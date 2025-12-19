# Data Analysis Report: Equity Market Structure

## 1. Dataset Overview
This analysis is based on daily OHLCV (Open, High, Low, Close, Volume) market data. The primary focus is on the **Adjusted Close** price to account for dividends and splits, ensuring accurate return calculations.

*   **Frequency:** Daily
*   **Data Type:** Time-Series Float
*   **Key Transformations:** Logarithmic Returns, Rolling Windows

## 2. Exploratory Data Analysis (EDA)

### 2.1 Return Distribution (The "Fat Tail" Problem)
Financial returns do not follow a perfect Normal (Gaussian) distribution. Our analysis confirms **Leptokurtosis** (Fat Tails).
*   **Observation:** Extreme events (market crashes) occur more frequently than a standard normal distribution would predict.
*   **Implication:** Linear models assuming normality will underestimate risk. This justifies the use of **Non-Linear models like XGBoost**.

### 2.2 Volatility Clustering
We observe distinct regimes where high volatility events cluster together.
*   **Stationarity:** Raw prices are non-stationary (trends exist). Log returns are generally stationary but exhibit heteroskedasticity (variance changes over time).
*   **Feature Engineering:** This validates our decision to use multiple rolling volatility windows (5d, 21d, 63d) to capture these changing variance regimes.

## 3. Feature Correlation Analysis
A correlation heatmap was generated to check for multicollinearity among features.

| Feature Pair | Correlation | Interpretation |
| :--- | :--- | :--- |
| **SMA_21 vs SMA_63** | High (> 0.9) | Expected trend correlation. Tree-based models handle this well, but we use "Distance from SMA" to decorrelate. |
| **Vol_5d vs Vol_21d** | Moderate (0.6) | Short-term volatility spikes often lead long-term volatility. |
| **RSI vs Log_Ret** | Moderate (0.5) | Momentum is naturally tied to recent price direction. |

**Action:** We retain correlated features as GBDTs can select the most informative split, but we prioritize relative metrics (e.g., `Dist_SMA`) over absolute values.

## 4. Target Variable Distribution (Class Imbalance)
The "Crisis" target (defined as a -10% drop over the next 20 days) creates a highly imbalanced dataset.

*   **Normal Days (Class 0):** ~90-95% of the data.
*   **Crisis Days (Class 1):** ~5-10% of the data.

**Handling Strategy:**
1.  **Evaluation Metric:** We cannot use simple Accuracy. We must use **ROC-AUC** and **Precision-Recall** curves.
2.  **Model Tuning:** The XGBoost `scale_pos_weight` parameter will be tuned to penalize false negatives (missing a crisis) more than false positives.

## 5. Key Insights for Modeling
1.  **Lagged Relationships:** Volatility spikes often *precede* price crashes.
2.  **Regime Persistence:** Once a crisis starts, it tends to persist. The model should account for recent state history.
3.  **Noise vs. Signal:** Daily returns are noisy; rolling averages (Trends) provide a cleaner signal for regime detection.