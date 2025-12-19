marknet-crisis-predictio/
├── .github/
│   └── workflows/          # CI/CD for automated testing
├── data/
│   ├── raw/                # Original datasets (e.g., OHLCV csvs)
│   └── processed/          # Cleaned data for modeling
├── notebooks/
│   ├── 01_predict_stock_market_crashes.ipynb
│   └── 02_equity_market_crisis_regime_gbdt.ipynb
├── src/                    # Modularized python code
│   ├── __init__.py
│   ├── data_loader.py      # Functions to fetch/clean data
│   ├── features.py         # Technical indicator generation
│   └── models.py           # GBDT and prediction model classes
├── docs/                   # Documentation for just-the-docs
│   ├── index.md
│   └── methodology.md
├── .gitignore
├── README.md
└── requirements.txt


Equity-Market-Crisis-Prediction-GBDT/
├── .gitignore
├── LICENSE
├── README.md                 <-- The main documentation
├── requirements.txt          <-- Python dependencies
├── data/
│   ├── raw/                  <-- Original market data
│   └── processed/            <-- Cleaned datasets with features
├── notebooks/                <-- Your uploaded analysis files
│   ├── 01_data_preprocessing.ipynb  (renamed from gbdt1)
│   ├── 02_feature_engineering.ipynb (renamed from gbdt2)
│   ├── 03_model_training.ipynb      (renamed from gbdt3)
│   └── 04_evaluation_metrics.ipynb  (renamed from gbdt6)
├── src/                      <-- Modularized code
│   ├── __init__.py
│   ├── data_loader.py
│   └── features.py
└── images/                   <-- For README visualizations