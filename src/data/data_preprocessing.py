import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

nltk.download('wordnet')
nltk.download('stopwords')

logger = logging.getLogger('Data Preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('WARNING')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_path: str) -> tuple:
    try:
        train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
        test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
        logger.debug('Data Import Successful')
        return train_df, test_df
    except FileNotFoundError as e:
        logger.error(f"Error: File not found in path {data_path}. Details: {e}")
        raise
    except pd.errors.EmptyDataError:
        logger.error("Error: One of the CSV files is empty.")
        raise
    except pd.errors.ParserError:
        logger.error("Error: Error parsing one of the CSV files.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data: {e}")
        raise

def data_impute(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    try:
        train_df.fillna('', inplace=True)
        test_df.fillna('', inplace=True)
        logger.debug('Data Imputation done')
        return train_df, test_df
    except Exception as e:
        logger.error(f"An unexpected error occurred while imputing data: {e}")
        raise

def lemmatization(text: str) -> str:
    try:
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(y) for y in text]
        return " ".join(text)
    except Exception as e:
        logger.error(f"An unexpected error occurred during lemmatization: {e}")
        raise

def remove_stopwords(text: str) -> str:
    try:
        stop_words = set(stopwords.words("english"))
        text = [i for i in str(text).split() if i not in stop_words]
        return " ".join(text)
    except Exception as e:
        logger.error(f"An unexpected error occurred while removing stopwords: {e}")
        raise

def removing_numbers(text: str) -> str:
    try:
        text = ''.join([i for i in text if not i.isdigit()])
        return text
    except Exception as e:
        logger.error(f"An unexpected error occurred while removing numbers: {e}")
        raise

def lower_case(text: str) -> str:
    try:
        text = text.split()
        text = [y.lower() for y in text]
        return " ".join(text)
    except Exception as e:
        logger.error(f"An unexpected error occurred while converting to lower case: {e}")
        raise

def removing_punctuations(text: str) -> str:
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text)
        text = " ".join(text.split())
        return text.strip()
    except Exception as e:
        logger.error(f"An unexpected error occurred while removing punctuations: {e}")
        raise

def removing_urls(text: str) -> str:
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logger.error(f"An unexpected error occurred while removing URLs: {e}")
        raise

# def remove_small_sentences(df):
#     try:
#         for i in range(len(df)):
#             if len(df.text.iloc[i].split()) < 3:
#                 df.text.iloc[i] = np.nan
#     except KeyError:
#         print("Error: 'text' column not found in the dataframe.")
#         raise
#     except Exception as e:
#         print(f"An unexpected error occurred while removing small sentences: {e}")
#         raise

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.content = df.content.apply(lambda content: lower_case(content))
        logger.debug('lowercasing done')
        df.content = df.content.apply(lambda content: remove_stopwords(content))
        logger.debug('Stopwords removal successful')
        df.content = df.content.apply(lambda content: removing_numbers(content))
        logger.debug('digit removal successful')
        df.content = df.content.apply(lambda content: removing_punctuations(content))
        logger.debug('punctuation removal successful')
        df.content = df.content.apply(lambda content: removing_urls(content))
        logger.debug('URL removal successful')
        df.content = df.content.apply(lambda content: lemmatization(content))
        logger.debug('Lemmatization Successful')
        return df
    except KeyError:
        logger.error("Error: 'content' column not found in the dataframe.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while normalizing text: {e}")
        raise

def save_processed_data(train_df: pd.DataFrame, test_df: pd.DataFrame, data_path: str) -> None:
    try:
        data_path = os.path.join(data_path, 'interim')
        os.makedirs(data_path, exist_ok=True)
        train_df.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        test_df.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)
        logger.debug('preprocess data exported to CSV')
    except IOError as e:
        logger.error(f"Error: An I/O error occurred while saving processed data: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving processed data: {e}")
        raise

def main():
    try:
        train_df, test_df = load_data(data_path='data/raw')
        train_df, test_df = data_impute(train_df, test_df)
        train_process_data = normalize_text(train_df)
        test_process_data = normalize_text(test_df)
        save_processed_data(train_process_data, test_process_data, data_path='data')
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")

if __name__ == '__main__':
    main()
