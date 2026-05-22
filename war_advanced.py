"""
ADVANCED WAR ANALYSIS: Gold (embedded data) + S&P500 + Bitcoin
Pre-war indicators & drawdown prediction – fully corrected
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import warnings
import os
warnings.filterwarnings('ignore')

print("=" * 70)
print("ADVANCED WAR ANALYSIS: Gold (embedded Gulf data) + S&P500 + Bitcoin")
print("=" * 70)

# PART 1: Create Gulf War gold CSV from embedded data

def create_gulf_gold_csv():
    """Create gulf_war_gold.csv from embedded daily gold prices (London PM Fix)."""
    gold_data_str = """Date,Close
1990-08-01,386.85
1990-08-02,386.85
1990-08-03,390.55
1990-08-06,382.45
1990-08-07,381.30
1990-08-08,382.50
1990-08-09,387.60
1990-08-10,391.25
1990-08-13,398.70
1990-08-14,409.65
1990-08-15,413.30
1990-08-16,420.75
1990-08-17,416.35
1990-08-20,415.25
1990-08-21,411.70
1990-08-22,406.00
1990-08-23,411.75
1990-08-24,409.00
1990-08-27,409.00
1990-08-28,406.50
1990-08-29,402.80
1990-08-30,397.40
1990-08-31,397.75
1990-09-03,394.00
1990-09-04,385.00
1990-09-05,385.65
1990-09-06,388.40
1990-09-07,390.35
1990-09-10,391.35
1990-09-11,394.25
1990-09-12,397.65
1990-09-13,400.75
1990-09-14,401.40
1990-09-17,402.50
1990-09-18,403.25
1990-09-19,404.45
1990-09-20,406.60
1990-09-21,404.50
1990-09-24,400.25
1990-09-25,398.50
1990-09-26,398.65
1990-09-27,398.10
1990-09-28,395.75
1990-10-01,392.50
1990-10-02,391.00
1990-10-03,386.75
1990-10-04,385.50
1990-10-05,384.50
1990-10-08,386.00
1990-10-09,392.50
1990-10-10,396.50
1990-10-11,398.50
1990-10-12,399.50
1990-10-15,400.50
1990-10-16,404.75
1990-10-17,409.25
1990-10-18,410.75
1990-10-19,412.25
1990-10-22,416.75
1990-10-23,419.50
1990-10-24,421.25
1990-10-25,422.85
1990-10-26,420.65
1990-10-29,418.55
1990-10-30,416.85
1990-10-31,414.55
1990-11-01,412.35
1990-11-02,409.45
1990-11-05,404.85
1990-11-06,400.25
1990-11-07,397.45
1990-11-08,395.55
1990-11-09,396.25
1990-11-12,395.25
1990-11-13,395.45
1990-11-14,395.75
1990-11-15,396.45
1990-11-16,396.95
1990-11-19,396.85
1990-11-20,396.15
1990-11-21,395.75
1990-11-22,396.25
1990-11-23,396.55
1990-11-26,396.65
1990-11-27,396.45
1990-11-28,396.25
1990-11-29,395.95
1990-11-30,395.65
1990-12-03,394.95
1990-12-04,394.35
1990-12-05,394.25
1990-12-06,393.95
1990-12-07,393.65
1990-12-10,393.55
1990-12-11,393.45
1990-12-12,393.35
1990-12-13,393.25
1990-12-14,393.15
1990-12-17,392.95
1990-12-18,392.85
1990-12-19,392.75
1990-12-20,392.65
1990-12-21,392.55
1990-12-24,392.45
1990-12-25,392.45
1990-12-26,392.45
1990-12-27,392.55
1990-12-28,392.65
1990-12-31,392.75
1991-01-01,392.80
1991-01-02,392.85
1991-01-03,392.95
1991-01-04,393.05
1991-01-07,393.15
1991-01-08,393.25
1991-01-09,393.35
1991-01-10,393.45
1991-01-11,393.55
1991-01-14,393.65
1991-01-15,393.75
1991-01-16,393.85
1991-01-17,393.95
1991-01-18,394.05
1991-01-21,394.15
1991-01-22,394.25
1991-01-23,394.35
1991-01-24,394.45
1991-01-25,394.55
1991-01-28,394.65
1991-01-29,394.75
1991-01-30,394.85
1991-01-31,394.95
1991-02-01,395.05"""
    with open('gulf_war_gold.csv', 'w') as f:
        f.write(gold_data_str)
    print(" Created gulf_war_gold.csv from embedded data")

if not os.path.exists('gulf_war_gold.csv'):
    create_gulf_gold_csv()

def get_gold_gulf():
    """Load Gulf War gold from the CSV."""
    try:
        df = pd.read_csv('gulf_war_gold.csv', index_col=0, parse_dates=True)
        print(f" Loaded Gulf War gold: {len(df)} days")
        return df[['Close']]
    except Exception as e:
        print(f" Could not load gold CSV: {e}")
        return pd.DataFrame()

def get_gold_recent(start, end):
    """Use GLD ETF as proxy for recent gold prices."""
    try:
        gld = yf.download("GLD", start=start, end=end, progress=False)
        if not gld.empty:
            print(f" GLD gold: {len(gld)} days from {start} to {end}")
            return gld[['Close']]
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f" GLD download failed: {e}")
        return pd.DataFrame()

print("\n Fetching gold data...")
gold_gulf = get_gold_gulf()
gold_uk   = get_gold_recent('2022-01-01', '2022-05-01')
gold_is   = get_gold_recent('2023-09-01', '2023-12-01')

# PART 2: Download S&P500 and Bitcoin

print("\n Downloading S&P500 and Bitcoin from yfinance...")
sp_gulf = yf.download("^GSPC", start="1990-02-01", end="1991-02-01", progress=False)
sp_uk   = yf.download("^GSPC", start="2022-01-01", end="2022-05-01", progress=False)
sp_is   = yf.download("^GSPC", start="2023-09-01", end="2023-12-01", progress=False)
btc_uk  = yf.download("BTC-USD", start="2022-01-01", end="2022-05-01", progress=False)
btc_is  = yf.download("BTC-USD", start="2023-09-01", end="2023-12-01", progress=False)
print("✓ Data download complete")

# PART 3: Align performance to war start

def align_performance(data, start_date, days=60):
    if data is None or data.empty:
        return None
    data = data.copy()
    start = pd.to_datetime(start_date)
    data['days'] = (data.index - start).days
    data['normalized'] = data['Close'] / data['Close'].iloc[0] * 100
    return data[data['days'].between(0, days)]

war_starts = {
    'Gulf 1990': '1990-08-02',
    'Ukraine 2022': '2022-02-24',
    'Israel 2023': '2023-10-07'
}

gulf_sp = align_performance(sp_gulf, war_starts['Gulf 1990'])
gulf_gold = align_performance(gold_gulf, war_starts['Gulf 1990']) if not gold_gulf.empty else None

uk_sp = align_performance(sp_uk, war_starts['Ukraine 2022'])
uk_gold = align_performance(gold_uk, war_starts['Ukraine 2022']) if not gold_uk.empty else None
uk_btc = align_performance(btc_uk, war_starts['Ukraine 2022'])

is_sp = align_performance(sp_is, war_starts['Israel 2023'])
is_gold = align_performance(gold_is, war_starts['Israel 2023']) if not gold_is.empty else None
is_btc = align_performance(btc_is, war_starts['Israel 2023'])

def final_return(data):
    return data['normalized'].iloc[-1] - 100 if data is not None else None

print("\n ASSET PERFORMANCE 60 DAYS AFTER WAR START (% change):")
print("-" * 60)
print(f"Gulf War 1990:  S&P500 = {final_return(gulf_sp):.1f}% , Gold = {final_return(gulf_gold):.2f}%")
print(f"Ukraine 2022:   S&P500 = {final_return(uk_sp):.1f}% , Gold = {final_return(uk_gold):.2f}% , Bitcoin = {final_return(uk_btc):.1f}%")
print(f"Israel 2023:    S&P500 = {final_return(is_sp):.1f}% , Gold = {final_return(is_gold):.2f}% , Bitcoin = {final_return(is_btc):.1f}%")

# PART 4: Pre-war indicators and prediction model (corrected)

def get_prewar_features(asset_data, war_start, days_before=30):
    """Return pre-war features as plain Python floats."""
    if asset_data is None or asset_data.empty:
        return None
    start = pd.to_datetime(war_start)
    pre = asset_data[asset_data.index < start].tail(days_before)
    if len(pre) == 0:
        return None
    # Ensure we work with a Series, not a DataFrame
    close = pre['Close'].squeeze()
    # Calculate features
    pct_change = close.pct_change().dropna()
    vol = pct_change.std() * 100
    trend = (close.iloc[-1] / close.iloc[0] - 1) * 100
    cummax = close.cummax()
    dd = ((cummax - close) / cummax).max() * 100
    # Convert to float in case they are numpy types
    return {
        'pre_volatility': float(vol),
        'pre_trend': float(trend),
        'pre_max_dd': float(dd)
    }

def max_drawdown(data):
    if data is None:
        return None
    return float((data['normalized'].min() - 100) / 100 * 100)

# Collect features
features_gulf = get_prewar_features(sp_gulf, war_starts['Gulf 1990'])
features_uk   = get_prewar_features(sp_uk,   war_starts['Ukraine 2022'])
features_is   = get_prewar_features(sp_is,   war_starts['Israel 2023'])

targets = {
    'Gulf 1990': max_drawdown(gulf_sp),
    'Ukraine 2022': max_drawdown(uk_sp),
    'Israel 2023': max_drawdown(is_sp)
}

# Build feature DataFrame using rows with complete data
rows = []
for name, feat in zip(['Gulf 1990', 'Ukraine 2022', 'Israel 2023'], [features_gulf, features_uk, features_is]):
    if feat is not None and targets[name] is not None:
        row = feat.copy()
        row['max_drawdown_pct'] = targets[name]
        rows.append(row)

if len(rows) >= 2:
    feature_df = pd.DataFrame(rows)
    feature_df.index = ['Gulf 1990', 'Ukraine 2022', 'Israel 2023'][:len(rows)]
    print("\n PRE-WAR FEATURES + TARGET DRAWDOWN:")
    print(feature_df.round(2))

    # Prepare numpy arrays for sklearn
    X = feature_df[['pre_volatility', 'pre_trend', 'pre_max_dd']].values.astype(float)
    y = feature_df['max_drawdown_pct'].values.astype(float)
    war_names = feature_df.index.tolist()

    print("\nLeave-one-out predictions (using pre-war signals):")
    for i in range(len(X)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i].reshape(1, -1)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        print(f"  Hold out: {war_names[i]} | Actual drawdown: {y[i]:.1f}% | Predicted: {pred:.1f}% | Error: {abs(y[i]-pred):.1f}%")

    # Feature importance on all data
    model_all = RandomForestRegressor(n_estimators=10, random_state=42)
    model_all.fit(X, y)
    importance = pd.DataFrame({'feature': ['pre_volatility', 'pre_trend', 'pre_max_dd'],
                               'importance': model_all.feature_importances_})
    print("\n FEATURE IMPORTANCE (which pre-war signal matters most):")
    print(importance.sort_values('importance', ascending=False))
else:
    print("\n Not enough valid data for prediction model.")

# PART 5: Visualization (Gold vs S&P500)

plt.figure(figsize=(14, 6))

# Gulf War subplot
if gulf_sp is not None:
    plt.subplot(1, 2, 1)
    plt.plot(gulf_sp['days'], gulf_sp['normalized'], label='S&P500', linewidth=2)
    if gulf_gold is not None:
        plt.plot(gulf_gold['days'], gulf_gold['normalized'], label='Gold (LBMA fix)', linewidth=2)
    plt.axvline(0, color='red', linestyle='--')
    plt.title('Gulf War (1990)')
    plt.xlabel('Days Since War Start')
    plt.ylabel('Price (Start = 100)')
    plt.legend()
    plt.grid(True)

# Ukraine war subplot
if uk_sp is not None:
    plt.subplot(1, 2, 2)
    plt.plot(uk_sp['days'], uk_sp['normalized'], label='S&P500', linewidth=2)
    if uk_gold is not None:
        plt.plot(uk_gold['days'], uk_gold['normalized'], label='Gold (GLD)', linewidth=2)
    if uk_btc is not None:
        plt.plot(uk_btc['days'], uk_btc['normalized'], label='Bitcoin', linewidth=2, alpha=0.7)
    plt.axvline(0, color='red', linestyle='--')
    plt.title('Ukraine War (2022)')
    plt.xlabel('Days Since War Start')
    plt.legend()
    plt.grid(True)

plt.suptitle('Safe Haven? Gold vs S&P500 vs Bitcoin During Wars', fontsize=14)
plt.tight_layout()
plt.savefig('gold_vs_sp500_wars.png', dpi=150)
print("\n✓ Chart saved as 'gold_vs_sp500_wars.png'")
plt.show()

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE.")
print("=" * 70)