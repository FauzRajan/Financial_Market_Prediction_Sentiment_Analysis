import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import json
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

TICKERS = ['GOOGL', 'GOOG', 'AAPL', 'MSFT', 'AMZN']
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=5 * 365)

RAW_DATA_DIR = 'data/raw'
EXTERNAL_DATA_DIR = 'data/external'
SENTIMENT_TRAINING_DATA_DIR = 'data/sentiment_training'
PROCESSED_DATA_DIR = 'data/processed'

GOOGLE_DAILY_NEWS_FILE = os.path.join(RAW_DATA_DIR, 'Google_Daily_News.csv')
FINANCIAL_PHRASE_BANK_FILE = os.path.join(RAW_DATA_DIR, 'all-data.csv')
TWITTER_FINANCIAL_NEWS_TRAIN_FILE = os.path.join(RAW_DATA_DIR, 'sent_train.csv')
TWITTER_FINANCIAL_NEWS_VALID_FILE = os.path.join(RAW_DATA_DIR, 'sent_valid.csv')

def download_stock_prices(tickers, start_date, end_date, output_dir):
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if not data.empty:
                file_path = os.path.join(output_dir, f'{ticker}_prices.csv')
                data.to_csv(file_path)
            else:
                pass
        except Exception:
            pass

def load_google_daily_news(file_path, tickers_to_filter=None):
    try:
        df = pd.read_csv(file_path)
        df = df.rename(columns={'datetime': 'date', 'related': 'ticker', 'summary': 'text'})
        def safe_to_datetime(ts):
            try:
                return pd.to_datetime(ts, unit='s')
            except (ValueError, TypeError):
                return pd.to_datetime(ts, errors='coerce')
        df['date'] = df['date'].apply(safe_to_datetime)
        df.dropna(subset=['date'], inplace=True)
        df['date'] = df['date'].dt.normalize()
        if tickers_to_filter:
            if 'ticker' in df.columns and not df['ticker'].empty:
                df = df[df['ticker'].isin(tickers_to_filter)]
            else:
                pass
        df = df[['date', 'ticker', 'headline', 'text', 'url', 'source']]
        df.columns = ['date', 'ticker', 'headline', 'text', 'url', 'source']
        return df
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def load_financial_phrase_bank(file_path):
    try:
        # encoding and on_bad_lines necessary for Financial Phrase Bank CSV
        df = pd.read_csv(file_path, sep=',', header=None, names=['label', 'text'], encoding='latin-1', on_bad_lines='skip')
        df['label'] = df['label'].str.strip()
        return df
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def load_twitter_financial_news(train_file_path, valid_file_path):
    try:
        df_train = pd.read_csv(train_file_path)
        df_valid = pd.read_csv(valid_file_path)
        df = pd.concat([df_train, df_valid], ignore_index=True)
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        df['label'] = df['label'].map(sentiment_map)
        return df
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def run_data_collection():
    os.makedirs(EXTERNAL_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(SENTIMENT_TRAINING_DATA_DIR, exist_ok=True)
    download_stock_prices(TICKERS, START_DATE, END_DATE, EXTERNAL_DATA_DIR)
    google_daily_news_df = load_google_daily_news(GOOGLE_DAILY_NEWS_FILE, TICKERS)
    if not google_daily_news_df.empty:
        google_daily_news_df.to_json(os.path.join(PROCESSED_DATA_DIR, 'google_daily_news_unlabeled.json'), orient='records', date_format='iso', indent=4)
    financial_phrase_bank_df = load_financial_phrase_bank(FINANCIAL_PHRASE_BANK_FILE)
    if not financial_phrase_bank_df.empty:
        financial_phrase_bank_df.to_csv(os.path.join(SENTIMENT_TRAINING_DATA_DIR, 'financial_phrase_bank_labeled.csv'), index=False)
    twitter_financial_news_df = load_twitter_financial_news(
        TWITTER_FINANCIAL_NEWS_TRAIN_FILE,
        TWITTER_FINANCIAL_NEWS_VALID_FILE
    )
    if not twitter_financial_news_df.empty:
        twitter_financial_news_df.to_csv(os.path.join(SENTIMENT_TRAINING_DATA_DIR, 'twitter_financial_news_labeled.csv'), index=False)

if __name__ == "__main__":
    run_data_collection()