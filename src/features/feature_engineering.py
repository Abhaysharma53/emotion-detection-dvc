import numpy as np
import pandas as pd
import os
import yaml
import logging

from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger('Feature Engineering')
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

def load_params(url: str) -> dict:
    try:
        with open(url, 'r') as file:
            params = yaml.safe_load(file)
        max_feature = params.get('feature_engineering', {}).get('max_features', 1000)  # Default to 1000 if not specified
        logger.debug('parameter extracted from {}'.format(url))
        return max_feature
    except FileNotFoundError:
        logger.error(f"Error: The file {url} was not found.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except KeyError:
        logger.error("Error: 'feature_engineering' or 'max_features' key not found in params.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading parameters: {e}")
        raise

def load_data(data_path: str) -> tuple:
    try:
        train_df = pd.read_csv(os.path.join(data_path, 'train_processed.csv'))
        test_df = pd.read_csv(os.path.join(data_path, 'test_processed.csv'))
        train_df.fillna('', inplace=True)
        test_df.fillna('', inplace=True)
        logger.debug('Data Import successful')
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

def apply_count_vectorizer(train_df: pd.DataFrame, test_df: pd.DataFrame, max_feature: int) -> tuple:
    try:
        vectorizer = CountVectorizer(max_features=max_feature)
        X_train = train_df['content'].values
        Y_train = train_df['sentiment'].values

        X_test = test_df['content'].values
        Y_test = test_df['sentiment'].values
        
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        
        train_df_bow = pd.DataFrame(X_train_bow.toarray(), columns=vectorizer.get_feature_names_out())
        train_df_bow['label'] = Y_train

        test_df_bow = pd.DataFrame(X_test_bow.toarray(), columns=vectorizer.get_feature_names_out())
        test_df_bow['label'] = Y_test
        
        return train_df_bow, test_df_bow
    except KeyError as e:
        logger.error(f"Error: Column not found in dataframe: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while applying CountVectorizer: {e}")
        raise

def save_FE_data(train: pd.DataFrame, test: pd.DataFrame, data_path: str) -> None:
    try:
        data_path = os.path.join(data_path, 'processed')
        os.makedirs(data_path, exist_ok=True)
        train.to_csv(os.path.join(data_path, 'train_bow.csv'), index=False)
        test.to_csv(os.path.join(data_path, 'test_bow.csv'), index=False)
        logger.debug('Feature engineered data exported to csv')
    except IOError as e:
        logger.error(f"Error: An I/O error occurred while saving feature engineering data: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving feature engineering data: {e}")
        raise

def main():
    try:
        max_feature = load_params('params.yaml')
        train_data, test_data = load_data(data_path='data/interim')
        train_FE_data, test_FE_data = apply_count_vectorizer(train_data, test_data, max_feature)
        save_FE_data(train_FE_data, test_FE_data, data_path='data')
    except Exception as e:
        print(f"An error occurred in the main function: {e}")

if __name__ == '__main__':
    main()
