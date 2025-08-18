# FinSense: Stock Prediction via Financial News Sentiment

## Overview

This project is a machine learning pipeline that predicts daily stock returns by combining historical price data with financial news sentiment.

## Methodology

### 1. Data Collection (`src/data_collection.py`)

* **Data Sources:** Uses `yfinance` for 5 years of stock data and a custom dataset for financial news.

### 2. Sentiment Analysis (`src/sentiment_analysis.py`)

* **Custom DL Model:** A TensorFlow/Keras model trained on financial text is used to score news headlines, generating sentiment labels and probability scores.

### 3. Feature Engineering (`src/model_training.py`)

* **Feature Creation:** Combines stock price and sentiment data, creating features like lagged returns, sentiment, news volume, and technical indicators (RSI, MACD, etc.).

### 4. Predictive Modeling (`src/model_training.py`)

* **Model:** A RandomForestRegressor is used for prediction.

* **Evaluation:** `GridSearchCV` with `TimeSeriesSplit` (Walk-Forward Validation) ensures robust evaluation on unseen, chronologically split data.

## Key Findings & Results

The model demonstrated strong predictive performance.

* **Performance (on unseen test set):**

  * **MAE:** `0.0071`

  * **RMSE:** `0.0099`

  * **R-squared (R2):** `0.7341`

* **Feature Importance:** Technical indicators and lagged price features were most influential. Sentiment features had minimal importance due to data sparsity.

### Challenges & Solutions

1. **Data Parsing:** Handled inconsistent data types and CSV formats with explicit conversions.

2. **Sparse Sentiment Data:** Documented and engineered features to work around limited news data.

3. **Model Robustness:** Used `GridSearchCV` with `TimeSeriesSplit` for rigorous tuning and validation.

## Future Work

1. Integrate larger financial news datasets.

2. Explore advanced time-series models (LSTMs, Transformers).

3. Develop a robust backtesting framework.

## How to Run the Project

1. **Clone:** `git clone https://github.com/YourGitHubUsername/FinSense-Stock-Prediction.git`

2. **Setup:** `pip install -r requirements.txt` and place raw data files in `data/raw/`.

3. **Execute:**

   ```
   python src/data_collection.py
   python src/sentiment_analysis.py
   python src/model_training.py
   ```

## Project Structure

```
FinSense-Stock-Prediction/
├── data/
│   ├── raw/
│   ├── external/
│   └── processed/
├── src/
│   ├── data_collection.py
│   ├── sentiment_analysis.py
│   ├── model_training.py
│   └── utils.py
├── models/
├── reports/
│   └── figures/
├── .gitignore
├── README.md
└── requirements.txt
```

## Dependencies

Listed in `requirements.txt`.

## License

This project is open-sourced under the MIT License.
