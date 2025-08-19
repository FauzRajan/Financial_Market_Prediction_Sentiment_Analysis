import os
import sys
import nltk
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
    except Exception:
        pass
    try:
        nltk.download('wordnet', quiet=True)
    except Exception:
        pass
    try:
        nltk.download('punkt', quiet=True)
    except Exception:
        pass
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass
def run_script(script_name):
    script_path = os.path.join(SRC_DIR, script_name)
    if not os.path.exists(script_path):
        return False
    try:
        original_dir = os.getcwd()
        os.chdir(PROJECT_ROOT)
        module_name = f"src.{script_name.replace('.py', '')}"
        __import__(module_name)
        return True
    except Exception:
        return False
    finally:
        os.chdir(original_dir)
def main():
    download_nltk_data()
    if not run_script('data_collection.py'):
        return
    if not run_script('sentiment_analysis.py'):
        return
    if not run_script('model_training.py'):
        return
if __name__ == "__main__":
    main()