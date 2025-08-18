# FinSense: Stock Prediction via Financial News Sentiment

## Overview

This project is a machine learning pipeline that predicts daily stock returns by combining historical price data with financial news sentiment.

## Methodology

### 1. Data Collection (`src/data_collection.py`)

* Uses `yfinance` for 5 years of stock data and a custom dataset for financial news.

### 2. Sentiment Analysis (`src/sentiment_analysis.py`)

* A TensorFlow/Keras model trained on financial text is used to score news headlines, generating sentiment labels and probability scores.

### 3. Feature Engineering (`src/model_training.py`)

* Combines stock price and sentiment data, creating features like lagged returns, sentiment, news volume, and technical indicators (RSI, MACD, etc.).

### 4. Predictive Modeling (`src/model_training.py`)

* A RandomForestRegressor is used for prediction.

* **Evaluation:** `GridSearchCV` with `TimeSeriesSplit` (Walk-Forward Validation) for evaluation on unseen, chronologically split data.

* **Performance (on unseen test set):**

  * **MAE:** `0.0071`

  * **RMSE:** `0.0099`

  * **R-squared (R2):** `0.7341`

* **Feature Importance:** Technical indicators and lagged price features were most influential. Sentiment features had minimal importance due to data sparsity.

## Future Work

1. Integrate larger financial news datasets.

2. Explore advanced time-series models (LSTMs, Transformers).

3. Develop a robust backtesting framework.

## How to Run the Project

1. **Clone:** Clone the repository from Github

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
