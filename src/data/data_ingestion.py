import numpy as np
import pandas as pd
import os
import yaml
import logging

from sklearn.model_selection import train_test_split

logger = logging.getLogger('Data_Ingestion')
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
        test_size = params.get('data_ingestion', {}).get('test_size', 0.2)  # Default to 0.2 if not specified
        logger.debug("Parameter fetched from {}".format(url))
        return test_size
    except FileNotFoundError:
        logger.error(f"Error: The file {url} was not found.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except KeyError:
        logger.error("Error: 'data_ingestion' or 'test_size' key not found in params.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading parameters: {e}")
        raise

def load_dataset(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug('file fetched successfully from URL')
        return df
    except pd.errors.EmptyDataError:
        logger.error("Error: The data file is empty.")
        raise
    except pd.errors.ParserError:
        logger.error("Error: Error parsing the data file.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the dataset: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        logger.debug('tweet_id column dropped')
    except KeyError:
        logger.error("Error: 'tweet_id' column not found in the dataframe.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while dropping columns: {e}")
        raise
    
    final_df = df[df['sentiment'].isin(['neutral', 'sadness'])]
    
    if final_df.empty:
        logger.warning("Warning: No data left after filtering sentiments.")
    
    try:
        final_df['sentiment'].replace({'neutral': 0, 'sadness': 1}, inplace=True)
    except KeyError:
        logger.error("Error: 'sentiment' column not found in the dataframe.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while replacing sentiment values: {e}")
        raise
    
    return final_df

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, data_path: str) -> None:
    try:
        data_path = os.path.join(data_path, 'raw')
        os.makedirs(data_path, exist_ok=True)  # `exist_ok=True` prevents error if directory already exists
        train_df.to_csv(os.path.join(data_path, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(data_path, 'test.csv'), index=False)
        logger.debug('train and testing data saved successfully')
    except IOError as e:
        logger.error(f"Error: An I/O error occurred while saving data: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving data: {e}")
        raise

def main():
    try:
        test_size = load_params(url='params.yaml')
        df = load_dataset('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='data')
        logger.debug('Data Ingestion Successful')
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")

if __name__ == '__main__':
    main() 