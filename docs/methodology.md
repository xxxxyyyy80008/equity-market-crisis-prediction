# Methodology: Equity Market Crisis Regime Prediction

## 1. Problem Definition
The objective of this project is to predict the onset of "Crisis Regimes" in equity markets. A crisis regime is defined not just by a single day's drop, but by a sustained period of significant negative returns. By identifying these regimes early, risk management strategies can be adjusted to preserve capital.

We frame this as a **Binary Classification** problem:
- **Class 0 (Normal Regime):** Standard market behavior with typical volatility.
- **Class 1 (Crisis Regime):** Periods leading to significant drawdown.

## 2. Data Pipeline
The data ingestion process handles financial time-series data (OHLCV - Open, High, Low, Close, Volume).

### Preprocessing Steps:
1.  **Cleaning:** Handling missing values via forward-filling (to respect time continuity) or dropping.
2.  **Normalization:** Prices are converted to **Log Returns** to ensure stationarity, as raw prices are non-stationary.
3.  **Splitting:** We use a **Time Series Split** (not random shuffle) to prevent data leakage. The model is trained on past data and tested on future data.

## 3. Feature Engineering
We extract three categories of technical indicators to capture market dynamics. These features serve as the input vector $\mathbf{X}$ for our model.

| Category | Features | Rationale |
| :--- | :--- | :--- |
| **Volatility** | Rolling Std Dev (5d, 10d, 21d, 63d) | Crises are often preceded or accompanied by spikes in volatility. |
| **Momentum** | RSI (14d), Rate of Change (ROC) | Captures the speed of price changes; rapid selling pressure indicates stress. |
| **Trend** | Distance from SMA (21d, 63d) | Large deviations from long-term trends can signal mean reversion or panic. |

## 4. Target Variable Definition
Defining the "ground truth" for a crisis is critical. We use a **Forward-Looking Fixed Window** approach.

For any given day $t$, the target $y_t$ is defined as:

$$
y_t = 
\begin{cases} 
1 & \text{if } \sum_{i=1}^{20} r_{t+i} < -0.10 \\
0 & \text{otherwise}
\end{cases}
$$

*Where $r$ is the daily log return.*
*Meaning: If the cumulative return over the **next 20 days** drops below **-10%**, today is labeled as a pre-crisis day.*

## 5. Model Architecture: Gradient Boosted Decision Trees (GBDT)
We utilize **XGBoost (Extreme Gradient Boosting)** as our core classifier.

### Why GBDT?
1.  **Non-Linearity:** Can capture complex interactions between volatility and momentum.
2.  **Feature Importance:** Provides interpretability on which indicators signal a crisis.
3.  **Robustness:** Handles outliers better than linear models and reduces overfitting via regularization parameters ($\lambda$, $\alpha$).

### Hyperparameters
-   **Objective:** `binary:logistic` (Outputs probability of crisis).
-   **Evaluation Metric:** `logloss` (Penalizes confident wrong predictions).
-   **Max Depth:** Constrained (e.g., 3-5) to prevent memorizing noise.
-   **Learning Rate:** Low (e.g., 0.01 - 0.1) for better generalization.

## 6. Evaluation Strategy
Given the imbalance of the dataset (Crises are rare events), standard Accuracy is misleading. We focus on:

1.  **ROC-AUC Score:** Measures the model's ability to distinguish between classes across all thresholds.
2.  **Precision:** Out of all predicted crises, how many were real? (Minimizing False Alarms).
3.  **Recall:** Out of all real crises, how many did we catch? (Minimizing Missed Crises).
4.  **Confusion Matrix:** To visualize Type I vs Type II errors.