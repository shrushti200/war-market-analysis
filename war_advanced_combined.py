"""
WAR MARKET ANALYSIS
Historical event study: Gold, S&P 500, and Bitcoin

What this script does:
1. Downloads market prices from Yahoo Finance.
2. Compares returns over 7, 30, 60, and 90 trading observations.
3. Calculates maximum drawdown and pre-event indicators.
4. Creates event-study charts.
5. Saves results as CSV files.
6. Uses a simple baseline for small samples.
7. Runs exploratory Random Forest predictions only when enough
   events are available.

Important:
- Market movements around a conflict do not prove that the conflict
  caused those movements.
- Some assets did not exist or lack data for older events.
- This is research/education, not investment advice.
"""

from pathlib import Path
from datetime import date, timedelta
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. SETTINGS
# ============================================================

OUTPUT_DIR = Path("war_analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Trading observations AFTER the event's first available price.
HORIZONS = [7, 30, 60, 90]

# Number of previous trading observations used for pre-event features.
PRE_EVENT_DAYS = 60

# Require at least this many events before fitting a Random Forest.
MIN_EVENTS_FOR_MODEL = 8

# Events are dated by the selected event/announcement date.
# These dates are research choices, not claims that a conflict began
# in every sense on that exact date.
EVENTS = {
    "Gulf War": "1990-08-02",
    "Iraq War": "2003-03-20",
    "Crimea Crisis": "2014-02-20",
    "Iran-US Escalation": "2020-01-03",
    "Russia-Ukraine Invasion": "2022-02-24",
    "Israel-Hamas War": "2023-10-07",
}

TICKERS = {
    "S&P 500": "^GSPC",
    "Gold Futures": "GC=F",
    "Bitcoin": "BTC-USD",
}

print("=" * 75)
print("WAR MARKET ANALYSIS")
print("=" * 75)


# ============================================================
# 2. DOWNLOAD MARKET DATA
# ============================================================

def download_prices(ticker, start, end):
    """
    Download adjusted close prices where available.

    yfinance can return either ordinary columns or MultiIndex columns.
    This function handles both formats.
    """
    print(f"Downloading {ticker}...")

    try:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if data.empty:
            print(f"  WARNING: No data returned for {ticker}.")
            return pd.Series(dtype=float, name=ticker)

        close = data["Close"]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close = pd.to_numeric(close, errors="coerce").dropna()
        close.index = pd.to_datetime(close.index)

        # Remove timezone only if one exists.
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)

        close = close[~close.index.duplicated(keep="last")]
        close = close.sort_index()
        close.name = ticker

        print(
            f"  Retrieved {len(close)} observations: "
            f"{close.index.min().date()} to {close.index.max().date()}"
        )

        return close

    except Exception as exc:
        print(f"  WARNING: Could not download {ticker}: {exc}")
        return pd.Series(dtype=float, name=ticker)


# Start early enough to calculate pre-event features for the first event.
download_start = (
    pd.Timestamp(min(EVENTS.values())) - pd.Timedelta(days=180)
).strftime("%Y-%m-%d")

# yfinance's end date is exclusive, so use tomorrow.
download_end = (date.today() + timedelta(days=1)).isoformat()

price_data = {}

for asset_name, ticker in TICKERS.items():
    price_data[asset_name] = download_prices(
        ticker,
        download_start,
        download_end,
    )

available_series = [
    series.rename(asset)
    for asset, series in price_data.items()
    if not series.empty
]

if not available_series:
    raise RuntimeError(
        "No market data was downloaded. Check your internet connection "
        "and try again."
    )

# Explicitly sort the combined index to avoid a Pandas concatenation warning.
all_prices = pd.concat(
    available_series,
    axis=1,
    sort=False,
).sort_index()

all_prices.to_csv(OUTPUT_DIR / "downloaded_market_prices.csv")

print("\nSaved downloaded_market_prices.csv")


def calculate_return_pct(prices):
    """Calculate percentage return from the first to last price."""
    if prices is None or len(prices) < 2:
        return None

    start_price = prices.iloc[0]
    end_price = prices.iloc[-1]

    if start_price == 0:
        return None

    return ((end_price - start_price) / start_price) * 100



# ============================================================
# 3. EVENT-STUDY FUNCTIONS
# ============================================================

def get_event_window(
    series,
    event_date,
    observations,
    max_gap_days=7,
):
    """
    Select prices only if the first available observation is within
    max_gap_days of the event date.

    If an asset has no nearby data, return an empty Series.
    This prevents future prices from being incorrectly assigned
    to historical events.
    """
    if series is None or series.empty:
        return pd.Series(dtype=float)

    event_date = pd.Timestamp(event_date)

    prices = series.copy()
    prices.index = pd.to_datetime(prices.index)

    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    prices = prices.sort_index().dropna()

    # Only consider observations on or after the event.
    after_event = prices.loc[prices.index >= event_date]

    if after_event.empty:
        return pd.Series(dtype=float)

    # Reject data that begins too long after the event.
    first_date = after_event.index[0]
    gap_days = (first_date - event_date).days

    if gap_days > max_gap_days:
        return pd.Series(dtype=float)

    window = after_event.iloc[:observations + 1].copy()

    if len(window) < 2 or window.iloc[0] <= 0:
        return pd.Series(dtype=float)

    # Normalize the first available price to 100.
    normalized = window / window.iloc[0] * 100
    normalized.name = series.name


    return normalized

def calculate_max_drawdown_pct(window):
    """
    Calculate the largest peak-to-trough fall in a price window.
    Returns a negative percentage, or NaN when data is insufficient.
    """
    if window is None or len(window) < 2:
        return np.nan

    running_peak = window.cummax()
    drawdown = (window / running_peak - 1) * 100

    return float(drawdown.min())


# ============================================================
# 4. RETURNS AND DRAWDOWN FOR ALL EVENTS
# ============================================================

performance_rows = []
normalized_windows = {}

print("\n")
print("=" * 75)
print("EVENT PERFORMANCE")
print("=" * 75)

for event_name, event_date in EVENTS.items():

    print(f"\n{event_name} — {event_date}")

    normalized_windows[event_name] = {}

    for asset_name, series in price_data.items():

        normalized_windows[event_name][asset_name] = {}

        if series.empty:
            print(f"  {asset_name}: unavailable")
            continue

        for horizon in HORIZONS:

            window = get_event_window(
                series,
                event_date,
                horizon,
            )

            normalized_windows[event_name][asset_name][horizon] = window

            ret = calculate_return_pct(window)
            drawdown = calculate_max_drawdown_pct(window)

            if ret is None or pd.isna(ret):
                print(
                    f"  {asset_name}, {horizon} observations: "
                    "insufficient data"
                )
                continue

            print(
                f"  {asset_name:15s} | "
                f"{horizon:2d} observations | "
                f"Return {ret:+7.2f}% | "
                f"Max drawdown {drawdown:7.2f}%"
            )

            performance_rows.append({
                "event": event_name,
                "event_date": event_date,
                "asset": asset_name,
                "horizon_trading_observations": horizon,
                "actual_start_date": window.index[0].date().isoformat(),
                "actual_end_date": window.index[-1].date().isoformat(),
                "observations_available": len(window) - 1,
                "return_pct": round(ret, 4),
                "max_drawdown_pct": round(drawdown, 4),
            })


performance_df = pd.DataFrame(performance_rows)

performance_df.to_csv(
    OUTPUT_DIR / "event_performance.csv",
    index=False,
)

print("\nSaved event_performance.csv")


# ============================================================
# 5. PRE-EVENT INDICATORS
# ============================================================

def calculate_pre_event_features(
    series,
    event_date,
    observations=60,
):
    """
    Calculate indicators using prices strictly before the event date.

    Features:
    - pre_volatility_pct: standard deviation of daily returns, in percent
    - pre_trend_pct: total price change over the pre-event window
    - pre_max_drawdown_pct: maximum pre-event drawdown magnitude
    """
    if series is None or series.empty:
        return None

    event_date = pd.Timestamp(event_date)

    previous_prices = series.loc[
        series.index < event_date
    ].dropna().tail(observations + 1)

    if len(previous_prices) < 10:
        return None

    daily_returns = previous_prices.pct_change().dropna()

    volatility = daily_returns.std(ddof=1) * 100

    trend = (
        previous_prices.iloc[-1] / previous_prices.iloc[0] - 1
    ) * 100

    running_peak = previous_prices.cummax()

    pre_drawdown = (
        (previous_prices / running_peak - 1).min() * -100
    )

    return {
        "pre_volatility_pct": float(volatility),
        "pre_trend_pct": float(trend),
        "pre_max_drawdown_pct": float(pre_drawdown),
    }


# We use the S&P 500 as the consistent target series.
# Its history is longer than Bitcoin's or gold futures' history.
feature_rows = []

sp500 = price_data["S&P 500"]

for event_name, event_date in EVENTS.items():

    features = calculate_pre_event_features(
        sp500,
        event_date,
        PRE_EVENT_DAYS,
    )

    target_window = get_event_window(
        sp500,
        event_date,
        60,
    )

    target_drawdown = calculate_max_drawdown_pct(target_window)

    if features is None or np.isnan(target_drawdown):
        print(
            f"Skipping pre-event model row for {event_name}: "
            "insufficient S&P 500 data."
        )
        continue

    row = {
        "event": event_name,
        "event_date": event_date,
        **features,
        "target_max_drawdown_pct": target_drawdown,
    }

    feature_rows.append(row)


feature_df = pd.DataFrame(feature_rows)

feature_df.to_csv(
    OUTPUT_DIR / "pre_event_features.csv",
    index=False,
)

print("\n")
print("=" * 75)
print("PRE-EVENT INDICATORS")
print("=" * 75)

if feature_df.empty:
    print("No events had enough data for the pre-event analysis.")
else:
    print(feature_df.round(3).to_string(index=False))

print("\nSaved pre_event_features.csv")

# SECTION 6: BASELINE, BENCHMARK, AND OPTIONAL RANDOM FOREST
# --------------------------------------------------------

if len(feature_df) < 3:
    print("\nNot enough events to run predictive analysis.")
else:
    feature_columns = [
        "pre_volatility_pct",
        "pre_trend_pct",
        "pre_max_drawdown_pct",
    ]

    X = feature_df[feature_columns].copy()
    y = feature_df["target_max_drawdown_pct"].copy()

    # Leave-one-event-out median baseline

   
    baseline_rows = []

    for i in range(len(feature_df)):
        train_y = y.drop(y.index[i])
        actual = float(y.iloc[i])

        if len(train_y) == 0:
            continue

        prediction = float(train_y.median())
        error = abs(actual - prediction)

        baseline_rows.append({
            "event": feature_df["event"].iloc[i],
            "actual_max_drawdown_pp": actual,
            "baseline_prediction_pp": prediction,
            "baseline_absolute_error_pp": error,
        })

    baseline_df = pd.DataFrame(baseline_rows)

    baseline_mae = float(
        baseline_df["baseline_absolute_error_pp"].mean()
    )

    print(
        f"\nBaseline leave-one-event-out MAE: "
        f"{baseline_mae:.2f} percentage points"
    )

    # Zero-drawdown benchmark
    zero_predictions = np.zeros(len(y))
    zero_errors = np.abs(y.to_numpy() - zero_predictions)
    zero_mae = float(np.mean(zero_errors))

    benchmark_comparison_df = pd.DataFrame([
        {
            "model": "Median baseline (leave-one-event-out)",
            "mae_pp": baseline_mae,
        },
        {
            "model": "Zero-drawdown benchmark",
            "mae_pp": zero_mae,
        },
    ])

    benchmark_comparison_df.to_csv(
        OUTPUT_DIR / "benchmark_comparison.csv",
        index=False,
    )

    print("\nBenchmark comparison:")
    print(benchmark_comparison_df.to_string(index=False))
    print("\nSaved benchmark_comparison.csv")

    baseline_df.to_csv(
        OUTPUT_DIR / "baseline_predictions.csv",
        index=False,
    )
    print("\nSaved baseline_predictions.csv")


    # --------------------------------------------------------
    # Optional Random Forest
    # --------------------------------------------------------

    MIN_EVENTS_FOR_MODEL = 8

    if len(feature_df) >= MIN_EVENTS_FOR_MODEL:

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import LeaveOneOut
        from sklearn.metrics import mean_absolute_error

        loo = LeaveOneOut()
        model_rows = []

        for train_index, test_index in loo.split(X):

            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]

            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=3,
                min_samples_leaf=2,
                random_state=42,
            )

            model.fit(X_train, y_train)

            prediction = float(model.predict(X_test)[0])
            actual = float(y_test.iloc[0])

            model_rows.append({
                "event": feature_df["event"].iloc[test_index[0]],
                "actual_max_drawdown_pp": actual,
                "random_forest_prediction_pp": prediction,
                "absolute_error_pp": abs(actual - prediction),
            })

        model_df = pd.DataFrame(model_rows)

        model_df.to_csv(
            OUTPUT_DIR / "random_forest_predictions.csv",
            index=False,
        )

        rf_mae = float(
            mean_absolute_error(
                model_df["actual_max_drawdown_pp"],
                model_df["random_forest_prediction_pp"],
            )
        )

        print(
            f"\nRandom Forest leave-one-event-out MAE: "
            f"{rf_mae:.2f} percentage points"
        )

                # Feature importance is exploratory for a small dataset.
        full_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=2,
            random_state=42,
        )

        full_model.fit(X, y)

        importance_df = pd.DataFrame({
            "feature": feature_columns,
            "importance": full_model.feature_importances_,
        }).sort_values(
            "importance",
            ascending=False,
        )

        importance_df.to_csv(
            OUTPUT_DIR / "feature_importance.csv",
            index=False,
        )

        print("\nFeature importance:")
        print(importance_df.to_string(index=False))

# ============================================================
# 7. CHART: NORMALIZED PRICES FOR EACH EVENT
# ============================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(15, 14),
)

axes = axes.flatten()

asset_colors = {
    "S&P 500": "tab:blue",
    "Gold Futures": "tab:orange",
    "Bitcoin": "tab:green",
}

for ax, (event_name, event_date) in zip(axes, EVENTS.items()):

    plotted = False

    for asset_name in TICKERS:

        series = price_data[asset_name]

        if series.empty:
            continue

        window = get_event_window(
            series,
            event_date,
            60,
        )

        if len(window) < 2:
            continue

        # Use trading-observation number on the x-axis.
        x_values = np.arange(len(window))

        ax.plot(
            x_values,
            window.values,
            label=asset_name,
            linewidth=1.8,
            color=asset_colors[asset_name],
        )

        plotted = True

    ax.axhline(
        100,
        color="gray",
        linestyle=":",
        linewidth=1,
    )

    ax.set_title(f"{event_name} ({event_date})")
    ax.set_xlabel("Trading observations after event")
    ax.set_ylabel("Indexed price (start = 100)")
    ax.grid(True, alpha=0.3)

    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "Insufficient price data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )


fig.suptitle(
    "Market Performance Around Selected Geopolitical Events",
    fontsize=16,
)

fig.tight_layout(rect=[0, 0, 1, 0.97])

chart_path = OUTPUT_DIR / "event_study_chart.png"

fig.savefig(
    chart_path,
    dpi=160,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# 8. SUMMARY AND OUTPUTS
# ============================================================

print("\n")
print("=" * 75)
print("PROJECT COMPLETE")
print("=" * 75)

print(f"Output folder: {OUTPUT_DIR.resolve()}")

print("\nFiles created:")
print("  downloaded_market_prices.csv")
print("  event_performance.csv")
print("  pre_event_features.csv")
print("  baseline_predictions.csv (if enough events)")
print("  random_forest_predictions.csv (if enough events)")
print("  feature_importance.csv (if model was fitted)")
print("  event_study_chart.png")

print("\nResearch limitations:")
print("- The event list is selective and not a random sample.")
print("- Prices can move for reasons unrelated to geopolitical events.")
print("- Trading observations are not the same as calendar days.")
print("- Gold futures, Bitcoin, and the S&P 500 have different histories.")
print("- A small event sample cannot establish reliable predictive power.")
print("- Historical results do not guarantee future market performance.")


# ============================================================
# 9. ADDITIONAL PORTFOLIO ANALYSIS AND CHARTS
# ============================================================

print("\n")
print("=" * 75)
print("ADDITIONAL PORTFOLIO ANALYSIS")
print("=" * 75)

# ------------------------------------------------------------
# A. Relative performance compared with the S&P 500
# ------------------------------------------------------------

if not performance_df.empty:

    sp500_returns = performance_df.loc[
        performance_df["asset"] == "S&P 500",
        [
            "event",
            "horizon_trading_observations",
            "return_pct",
        ],
    ].rename(
        columns={"return_pct": "sp500_return_pct"}
    )

    relative_df = performance_df.merge(
        sp500_returns,
        on=["event", "horizon_trading_observations"],
        how="left",
    )

    # Difference in percentage points, not a compounded relative return.
    relative_df["relative_return_vs_sp500_pp"] = (
        relative_df["return_pct"] - relative_df["sp500_return_pct"]
    )

    relative_path = OUTPUT_DIR / "relative_performance_vs_sp500.csv"

    relative_df.to_csv(relative_path, index=False)

    print(f"Saved: {relative_path.name}")

    # --------------------------------------------------------
    # B. Average returns by asset and horizon
    # --------------------------------------------------------

    return_pivot = performance_df.pivot_table(
        index="horizon_trading_observations",
        columns="asset",
        values="return_pct",
        aggfunc="mean",
    )

    if not return_pivot.empty:

        ax = return_pivot.plot(
            kind="bar",
            figsize=(11, 6),
            width=0.8,
        )

        ax.set_title(
            "Average Returns by Asset and Trading Horizon"
        )
        ax.set_xlabel("Trading observations after event")
        ax.set_ylabel("Average return (%)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="Asset")
        ax.tick_params(axis="x", rotation=0)

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "asset_returns_by_horizon.png",
            dpi=160,
            bbox_inches="tight",
        )

        plt.close()

        print("Saved: asset_returns_by_horizon.png")

    # --------------------------------------------------------
    # C. Average maximum drawdown by asset and horizon
    # --------------------------------------------------------

    drawdown_pivot = performance_df.pivot_table(
        index="horizon_trading_observations",
        columns="asset",
        values="max_drawdown_pct",
        aggfunc="mean",
    )

    if not drawdown_pivot.empty:

        ax = drawdown_pivot.plot(
            kind="bar",
            figsize=(11, 6),
            width=0.8,
        )

        ax.set_title(
            "Average Maximum Drawdown by Asset and Horizon"
        )
        ax.set_xlabel("Trading observations after event")
        ax.set_ylabel("Average maximum drawdown (%)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="Asset")
        ax.tick_params(axis="x", rotation=0)

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "drawdown_comparison.png",
            dpi=160,
            bbox_inches="tight",
        )

        plt.close()

        print("Saved: drawdown_comparison.png")

else:
    print("Portfolio charts skipped: no performance data available.")


# ------------------------------------------------------------
# D. Visualize pre-event indicators
# ------------------------------------------------------------

if not feature_df.empty:

    indicator_columns = [
        ("pre_volatility_pct", "Pre-event volatility (%)"),
        ("pre_trend_pct", "Pre-event price trend (%)"),
        ("pre_max_drawdown_pct", "Pre-event drawdown magnitude (%)"),
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 12),
    )

    for ax, (column, title) in zip(axes, indicator_columns):

        ax.bar(
            feature_df["event"],
            feature_df[column],
        )

        ax.set_title(title)
        ax.set_ylabel("%")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle(
        "S&P 500 Indicators Before Selected Geopolitical Events",
        fontsize=15,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(
        OUTPUT_DIR / "pre_event_indicators.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved: pre_event_indicators.png")

else:
    print("Pre-event indicator chart skipped: no feature data available.")


print("\nAdditional analysis finished.")
print(f"All outputs are saved in: {OUTPUT_DIR.resolve()}")
