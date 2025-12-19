# Equity Market Crisis & Regime Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research-orange)

## 📉 Project Overview
This project applies advanced machine learning techniques to predict stock market crashes and identify volatile market regimes. By leveraging **Gradient Boosted Decision Trees (GBDT)** and time-series analysis, we aim to provide early warning signals for equity market crises.

## 🧠 Key Components

### 1. Stock Market Crash Prediction
*   **Objective:** Forecast major drawdown events in major indices (e.g., S&P 500).
*   **Approach:** Binary classification of forward returns using macroeconomic indicators and technical volatility measures.
*   **Notebook:** `notebooks/01_predict_stock_market_crashes.ipynb`

### 2. Crisis Regime Prediction (GBDT)
*   **Objective:** Classify market states (e.g., Low Volatility, High Volatility, Crisis).
*   **Model:** Gradient Boosted Decision Trees (XGBoost/LightGBM).
*   **Key Findings:** GBDT outperforms traditional logistic regression in capturing non-linear relationships in market microstructure data.
*   **Notebook:** `notebooks/02_equity_market_crisis_regime_gbdt.ipynb`

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/market-crisis-prediction.git
cd market-crisis-prediction
pip install -r requirements.txt