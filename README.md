

\### 1. Install dependencies



```bash

pip install yfinance pandas numpy matplotlib scikit-learn

```



\### 2. Run the analysis



```bash

python war\_advanced\_combined.py

```



The script downloads market data and saves the generated results in the configured output directory.



\## Main Outputs



\- `downloaded\_market\_prices.csv` — historical market prices.

\- `event\_performance.csv` — asset performance around selected events.

\- `pre\_event\_features.csv` — pre-event indicators and drawdown targets.

\- `baseline\_predictions.csv` — baseline prediction results.

\- `benchmark\_comparison.csv` — comparison of baseline and zero-drawdown benchmark errors.

\- `event\_study\_chart.png` — event-study visualization.

\- `relative\_performance\_vs\_sp500.csv` — relative asset performance.

\- `asset\_returns\_by\_horizon.png` — asset returns across analysis horizons.

\- `drawdown\_comparison.png` — drawdown comparisons.

\- `pre\_event\_indicators.png` — pre-event indicator visualizations.



\## Methodological Limitations



\- The dataset contains only six selected geopolitical events, which is too small to establish reliable predictive performance.

\- Market movements can be influenced by many factors beyond geopolitical events.

\- Some assets have shorter historical price records than others.

\- Event windows may overlap with unrelated economic shocks.

\- Baseline results should be interpreted as exploratory, not as evidence of a dependable forecasting model.



\## Disclaimer



This project is intended for educational and research purposes only. It does not provide investment advice or trading recommendations.



