import pandas as pd
import numpy as np
import os
import json
import joblib
import sys
from datetime import datetime
import warnings
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning, module="matplotlib")
EXTERNAL_DATA_DIR = 'data/external'
PROCESSED_DATA_DIR = 'data/processed'
MODELS_DIR = 'models'
REPORTS_FIGURES_DIR = 'reports/figures'
SENTIMENT_SCORED_NEWS_FILE = os.path.join(PROCESSED_DATA_DIR, 'google_daily_news_with_sentiment.json')
COMBINED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'combined_stock_sentiment_data.csv')
EQUITY_PREDICTION_MODEL_PATH = os.path.join(MODELS_DIR, 'equity_prediction_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
CORRELATION_HEATMAP_PATH = os.path.join(REPORTS_FIGURES_DIR, 'correlation_heatmap.png')
FEATURE_IMPORTANCE_PATH = os.path.join(REPORTS_FIGURES_DIR, 'feature_importance.png')
ACTUAL_VS_PREDICTED_PLOT_PATH = os.path.join(REPORTS_FIGURES_DIR, 'actual_vs_predicted_returns.png')
RESIDUALS_PLOT_PATH = os.path.join(REPORTS_FIGURES_DIR, 'residuals_plot.png')
TIME_SERIES_PREDICTION_PLOT_PATH = os.path.join(REPORTS_FIGURES_DIR, 'time_series_predictions.png')
TICKERS = ['GOOGL', 'GOOG', 'AAPL', 'MSFT', 'AMZN']
def load_stock_prices(tickers, external_data_dir):
    all_prices = []
    for ticker in tickers:
        file_path = os.path.join(external_data_dir, f'{ticker}_prices.csv')
        try:
            df = pd.read_csv(file_path)
            date_col_name = 'Date' if 'Date' in df.columns else df.columns[0]
            df['date'] = pd.to_datetime(df[date_col_name], errors='coerce').dt.normalize()
            if date_col_name != 'date':
                df = df.drop(columns=[date_col_name])
            df.dropna(subset=['date'], inplace=True)
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
            df = df.set_index('date').reset_index()
            df['ticker'] = ticker
            all_prices.append(df)
        except FileNotFoundError:
            pass
        except Exception:
            pass
    if all_prices:
        return pd.concat(all_prices, ignore_index=True)
    else:
        return pd.DataFrame()
def load_sentiment_scored_news(file_path):
    try:
        df = pd.read_json(file_path, orient='records')
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        return df
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()
def aggregate_daily_sentiment(sentiment_news_df):
    if sentiment_news_df.empty:
        return pd.DataFrame()
    sentiment_news_df['sentiment_score'] = sentiment_news_df['sentiment_label'].map({'negative': -1, 'neutral': 0, 'positive': 1})
    required_cols = ['date', 'ticker', 'sentiment_score', 'text', 'sentiment_prob_positive', 'sentiment_prob_negative', 'sentiment_prob_neutral', 'sentiment_label']
    for col in required_cols:
        if col not in sentiment_news_df.columns:
            return pd.DataFrame()
    aggregated_sentiment_df = sentiment_news_df.groupby(['date', 'ticker']).agg(
        avg_sentiment=('sentiment_score', 'mean'),
        news_volume=('text', 'count'),
        prob_positive_mean=('sentiment_prob_positive', 'mean'),
        prob_negative_mean=('sentiment_prob_negative', 'mean'),
        prob_neutral_mean=('sentiment_prob_neutral', 'mean'),
        sentiment_spread=('sentiment_prob_positive', lambda x: x.mean() - sentiment_news_df.loc[x.index, 'sentiment_prob_negative'].mean()),
        count_positive=('sentiment_label', lambda x: (x == 'positive').sum()),
        count_negative=('sentiment_label', lambda x: (x == 'negative').sum()),
        count_neutral=('sentiment_label', lambda x: (x == 'neutral').sum())
    ).reset_index()
    return aggregated_sentiment_df
def merge_data(stock_df, sentiment_df):
    if stock_df.empty or sentiment_df.empty:
        return pd.DataFrame()
    stock_df['date'] = pd.to_datetime(stock_df['date']).dt.normalize()
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.normalize()
    merged_df = pd.merge(stock_df, sentiment_df, on=['date', 'ticker'], how='left')
    merged_df['avg_sentiment'] = merged_df['avg_sentiment'].fillna(0)
    merged_df['news_volume'] = merged_df['news_volume'].fillna(0)
    merged_df['prob_positive_mean'] = merged_df['prob_positive_mean'].fillna(1/3)
    merged_df['prob_negative_mean'] = merged_df['prob_negative_mean'].fillna(1/3)
    merged_df['prob_neutral_mean'] = merged_df['prob_neutral_mean'].fillna(1/3)
    merged_df['sentiment_spread'] = merged_df['sentiment_spread'].fillna(0)
    merged_df['count_positive'] = merged_df['count_positive'].fillna(0)
    merged_df['count_negative'] = merged_df['count_negative'].fillna(0)
    merged_df['count_neutral'] = merged_df['count_neutral'].fillna(0)
    merged_df['Close'] = pd.to_numeric(merged_df['Close'], errors='coerce')
    merged_df.dropna(subset=['Close'], inplace=True)
    merged_df['Daily_Return'] = merged_df.groupby('ticker')['Close'].pct_change()
    merged_df.dropna(subset=['Daily_Return'], inplace=True)
    return merged_df
def engineer_features(df):
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
    df['avg_sentiment_lag1'] = df.groupby('ticker')['avg_sentiment'].shift(1).fillna(0)
    df['avg_sentiment_lag3'] = df.groupby('ticker')['avg_sentiment'].shift(3).fillna(0)
    df['avg_sentiment_lag5'] = df.groupby('ticker')['avg_sentiment'].shift(5).fillna(0)
    df['prob_positive_mean_lag1'] = df.groupby('ticker')['prob_positive_mean'].shift(1).fillna(1/3)
    df['prob_negative_mean_lag1'] = df.groupby('ticker')['prob_negative_mean'].shift(1).fillna(1/3)
    df['prob_neutral_mean_lag1'] = df.groupby('ticker')['prob_neutral_mean'].shift(1).fillna(1/3)
    df['sentiment_spread_lag1'] = df.groupby('ticker')['sentiment_spread'].shift(1).fillna(0)
    df['count_positive_lag1'] = df.groupby('ticker')['count_positive'].shift(1).fillna(0)
    df['count_negative_lag1'] = df.groupby('ticker')['count_negative'].shift(1).fillna(0)
    df['count_neutral_lag1'] = df.groupby('ticker')['count_neutral'].shift(1).fillna(0)
    df['Daily_Return_lag1'] = df.groupby('ticker')['Daily_Return'].shift(1).fillna(0)
    df['Daily_Return_lag3_avg'] = df.groupby('ticker')['Daily_Return'].shift(1).rolling(window=3).mean().fillna(0)
    df['Daily_Return_lag5_avg'] = df.groupby('ticker')['Daily_Return'].shift(1).rolling(window=5).mean().fillna(0)
    df['Rolling_Volatility_5d'] = df.groupby('ticker')['Daily_Return'].rolling(window=5).std().reset_index(level=0, drop=True).fillna(0)
    df['news_volume_lag1'] = df.groupby('ticker')['news_volume'].shift(1).fillna(0)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df.dropna(subset=['Close', 'Volume'], inplace=True)
    def add_technical_indicators(data):
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
        return data
    df = df.groupby('ticker').apply(add_technical_indicators).reset_index(drop=True)
    df['sentiment_volume_interaction'] = df['avg_sentiment_lag1'] * df['news_volume_lag1']
    df['positive_volume_interaction'] = df['prob_positive_mean_lag1'] * df['news_volume_lag1']
    df['negative_volume_interaction'] = df['prob_negative_mean_lag1'] * df['news_volume_lag1']
    df['spread_volume_interaction'] = df['sentiment_spread_lag1'] * df['news_volume_lag1']
    df.dropna(inplace=True)
    return df
def analyze_correlation(df, features, target, output_path):
    os.makedirs(REPORTS_FIGURES_DIR, exist_ok=True)
    data_for_corr = df[features + [target]].copy()
    correlation_matrix = data_for_corr.corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
def train_and_evaluate_model(df, features, target, model_path, scaler_path, feature_importance_path, actual_vs_predicted_path, residuals_plot_path, time_series_plot_path):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_FIGURES_DIR, exist_ok=True)
    tscv = TimeSeriesSplit(n_splits=5)
    X = df[features]
    y = df[target]
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=tscv, scoring=mae_scorer, n_jobs=-1)
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test_full)
    best_model.fit(X_train_scaled, y_train_full)
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    y_pred = best_model.predict(X_test_scaled)
    feature_importances = best_model.feature_importances_
    features_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=features_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(feature_importance_path)
    plt.close()
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test_full, y_pred, alpha=0.5)
    plt.plot([y_test_full.min(), y_test_full.max()], [y_test_full.min(), y_test_full.max()], '--r', lw=2)
    plt.title('Actual vs. Predicted Daily Returns')
    plt.xlabel('Actual Daily Return')
    plt.ylabel('Predicted Daily Return')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(actual_vs_predicted_path)
    plt.close()
    residuals = y_test_full - y_pred
    plt.figure(figsize=(12, 6))
    sns.histplot(residuals, bins=50, kde=True)
    plt.title('Residuals Distribution')
    plt.xlabel('Residuals (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(residuals_plot_path)
    plt.close()
    test_dates = df.loc[y_test_full.index, 'date']
    plt.figure(figsize=(15, 7))
    plt.plot(test_dates, y_test_full.values, label='Actual Returns', color='blue', alpha=0.7)
    plt.plot(test_dates, y_pred, label='Predicted Returns', color='red', linestyle='--')
    plt.title('Time Series Prediction of Daily Returns')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(time_series_plot_path)
    plt.close()
def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_FIGURES_DIR, exist_ok=True)
    stock_df = load_stock_prices(TICKERS, EXTERNAL_DATA_DIR)
    sentiment_news_df = load_sentiment_scored_news(SENTIMENT_SCORED_NEWS_FILE)
    if not stock_df.empty:
        if not sentiment_news_df.empty:
            aggregated_sentiment_df = aggregate_daily_sentiment(sentiment_news_df)
            combined_df = merge_data(stock_df, aggregated_sentiment_df)
            combined_df.to_csv(COMBINED_DATA_FILE, index=False)
            if not combined_df.empty:
                engineered_df = engineer_features(combined_df)
                if not engineered_df.empty:
                    final_features = [
                        'avg_sentiment_lag1', 'avg_sentiment_lag3', 'avg_sentiment_lag5',
                        'prob_positive_mean_lag1', 'prob_negative_mean_lag1', 'prob_neutral_mean_lag1',
                        'sentiment_spread_lag1', 'count_positive_lag1', 'count_negative_lag1', 'count_neutral_lag1',
                        'Daily_Return_lag1', 'Daily_Return_lag3_avg', 'Daily_Return_lag5_avg',
                        'Rolling_Volatility_5d', 'news_volume_lag1',
                        'Open', 'High', 'Low', 'Volume',
                        'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
                        'sentiment_volume_interaction', 'positive_volume_interaction',
                        'negative_volume_interaction', 'spread_volume_interaction'
                    ]
                    final_features = [f for f in final_features if f in engineered_df.columns]
                    analyze_correlation(engineered_df, final_features.copy(), 'Daily_Return', CORRELATION_HEATMAP_PATH)
                    train_and_evaluate_model(
                        engineered_df, final_features, 'Daily_Return',
                        EQUITY_PREDICTION_MODEL_PATH, SCALER_PATH, FEATURE_IMPORTANCE_PATH,
                        ACTUAL_VS_PREDICTED_PLOT_PATH, RESIDUALS_PLOT_PATH, TIME_SERIES_PREDICTION_PLOT_PATH
                    )
if __name__ == "__main__":
    main()