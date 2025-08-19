import pandas as pd
import numpy as np
import os
import json
import pickle
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import preprocess_text
SENTIMENT_TRAINING_DATA_DIR = 'data/sentiment_training'
PROCESSED_DATA_DIR = 'data/processed'
MODELS_DIR = 'models'
REPORTS_FIGURES_DIR = 'reports/figures'
FINANCIAL_PHRASE_BANK_LABELED_FILE = os.path.join(SENTIMENT_TRAINING_DATA_DIR, 'financial_phrase_bank_labeled.csv')
TWITTER_FINANCIAL_NEWS_LABELED_FILE = os.path.join(SENTIMENT_TRAINING_DATA_DIR, 'twitter_financial_news_labeled.csv')
GOOGLE_DAILY_NEWS_FILE = os.path.join(PROCESSED_DATA_DIR, 'google_daily_news_unlabeled.json')
SENTIMENT_MODEL_PATH = os.path.join(MODELS_DIR, 'sentiment_model.h5')
TOKENIZER_PATH = os.path.join(MODELS_DIR, 'tokenizer.pkl')
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
OOV_TOKEN = "<OOV>"
SENTIMENT_MAPPING = {'negative': 0, 'neutral': 1, 'positive': 2}
INVERSE_SENTIMENT_MAPPING = {0: 'negative', 1: 'neutral', 2: 'positive'}
def load_labeled_data(financial_phrase_bank_file, twitter_financial_news_file):
    all_labeled_data = []
    try:
        df_fpb = pd.read_csv(financial_phrase_bank_file)
        all_labeled_data.append(df_fpb)
    except FileNotFoundError:
        pass
    except Exception:
        pass
    try:
        df_twitter = pd.read_csv(twitter_financial_news_file)
        all_labeled_data.append(df_twitter)
    except FileNotFoundError:
        pass
    except Exception:
        pass
    if not all_labeled_data:
        return pd.DataFrame()
    combined_df = pd.concat(all_labeled_data, ignore_index=True)
    return combined_df
def preprocess_labeled_data(df):
    df['processed_text'] = df['text'].apply(preprocess_text)
    df.dropna(subset=['processed_text'], inplace=True)
    df = df[df['processed_text'].str.strip() != '']
    return df
def build_and_train_model(X_train, y_train, X_val, y_val, vocab_size, num_classes):
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=20,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    return model, history
def evaluate_model(model, X_test, y_test, label_encoder):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred_classes)
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Sentiment Model Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_plot_path = os.path.join(REPORTS_FIGURES_DIR, 'sentiment_confusion_matrix.png')
    plt.savefig(cm_plot_path)
    plt.close()
def plot_training_history(history, output_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    history_plot_path = os.path.join(output_dir, 'sentiment_training_history.png')
    plt.tight_layout()
    plt.savefig(history_plot_path)
    plt.close()
def plot_label_distribution(df, output_dir):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=df, order=df['label'].value_counts().index, palette='viridis')
    plt.title('Distribution of Sentiment Labels in Training Data')
    plt.xlabel('Sentiment Label')
    plt.ylabel('Count')
    label_dist_plot_path = os.path.join(output_dir, 'sentiment_label_distribution.png')
    plt.savefig(label_dist_plot_path)
    plt.close()
def apply_sentiment_model_to_unlabeled_news(model, tokenizer, unlabeled_news_file, output_file_path):
    try:
        unlabeled_df = pd.read_json(unlabeled_news_file, orient='records')
        unlabeled_df['processed_text'] = unlabeled_df['text'].apply(preprocess_text)
        unlabeled_df.dropna(subset=['processed_text'], inplace=True)
        unlabeled_df = unlabeled_df[unlabeled_df['processed_text'].str.strip() != '']
        if unlabeled_df.empty:
            return
        news_sequences = tokenizer.texts_to_sequences(unlabeled_df['processed_text'])
        news_padded_sequences = pad_sequences(news_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        sentiment_probabilities = model.predict(news_padded_sequences)
        sentiment_classes = np.argmax(sentiment_probabilities, axis=1)
        unlabeled_df['sentiment_label'] = [INVERSE_SENTIMENT_MAPPING[c] for c in sentiment_classes]
        for i, label in INVERSE_SENTIMENT_MAPPING.items():
            unlabeled_df[f'sentiment_prob_{label}'] = sentiment_probabilities[:, i]
        output_file_path = os.path.join(PROCESSED_DATA_DIR, 'google_daily_news_with_sentiment.json')
        unlabeled_df.to_json(output_file_path, orient='records', date_format='iso', indent=4)
    except FileNotFoundError:
        pass
    except Exception:
        pass
def run_sentiment_analysis():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(REPORTS_FIGURES_DIR, exist_ok=True)
    combined_df = load_labeled_data(FINANCIAL_PHRASE_BANK_LABELED_FILE, TWITTER_FINANCIAL_NEWS_LABELED_FILE)
    if combined_df.empty:
        return
    combined_df = preprocess_labeled_data(combined_df)
    plot_label_distribution(combined_df, REPORTS_FIGURES_DIR)
    label_encoder = LabelEncoder()
    combined_df['numerical_label'] = label_encoder.fit_transform(combined_df['label'])
    X = combined_df['processed_text']
    y = combined_df['numerical_label']
    X_train_text, X_val_text, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    tokenizer = Tokenizer(oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(X_train_text)
    vocab_size = len(tokenizer.word_index) + 1
    X_train_sequences = tokenizer.texts_to_sequences(X_train_text)
    X_val_sequences = tokenizer.texts_to_sequences(X_val_text)
    X_train_padded = pad_sequences(X_train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    X_val_padded = pad_sequences(X_val_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    num_classes = len(label_encoder.classes_)
    model, history = build_and_train_model(X_train_padded, y_train, X_val_padded, y_val, vocab_size, num_classes)
    plot_training_history(history, REPORTS_FIGURES_DIR)
    evaluate_model(model, X_val_padded, y_val, label_encoder)
    model.save(SENTIMENT_MODEL_PATH)
    with open(TOKENIZER_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    apply_sentiment_model_to_unlabeled_news(model, tokenizer, GOOGLE_DAILY_NEWS_FILE, os.path.join(PROCESSED_DATA_DIR, 'google_daily_news_with_sentiment.json'))
if __name__ == "__main__":
    run_sentiment_analysis()