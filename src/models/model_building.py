import numpy as np
import pandas as pd
import pickle
import yaml
import os
import logging

from sklearn.ensemble import GradientBoostingClassifier

logger = logging.getLogger('Model Building')
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
        n_estimator = params.get('model_building', {}).get('n_estimators', 100)  # Default to 100 if not specified
        learn_rate = params.get('model_building', {}).get('learning_rate', 0.1)  # Default to 0.1 if not specified
        logging.debug('parameters extracted from {}'.format(url))
        return n_estimator, learn_rate
    except FileNotFoundError:
        logger.error(f"Error: The file {url} was not found.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except KeyError:
        logger.error("Error: 'model_building', 'n_estimators', or 'learning_rate' key not found in params.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading parameters: {e}")
        raise

def load_feature(datapath: str) -> pd.DataFrame:
    try:
        train_data = pd.read_csv(os.path.join(datapath, 'train_bow.csv'))
        logger.debug('Feature Engineered data load successful')
        return train_data
    except FileNotFoundError as e:
        logger.error(f"Error: File not found in path {datapath}. Details: {e}")
        raise
    except pd.errors.EmptyDataError:
        logger.error("Error: The CSV file is empty.")
        raise
    except pd.errors.ParserError:
        logger.error("Error: Error parsing the CSV file.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading features: {e}")
        raise

def train_model(X_train: pd.DataFrame, Y_train: pd.DataFrame, n_estimator: int, learn_rate: float) -> GradientBoostingClassifier:
    try:
        clf = GradientBoostingClassifier(n_estimators=n_estimator, learning_rate=learn_rate)
        clf.fit(X_train, Y_train)
        logger.debug('Model fitting successful')
        return clf
    except ValueError as e:
        logger.error(f"Error: Value error during model training. Details: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while training the model: {e}")
        raise

def save_model(model: GradientBoostingClassifier) -> None:
    try:
        with open('models/model.pkl', 'wb') as file:
            pickle.dump(model, file)
            logger.debug('Model Serialization Successful')
    except IOError as e:
        logger.error(f"Error: An I/O error occurred while saving the model: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving the model: {e}")
        raise

def main():
    try:
        n_estimator, learn_rate = load_params('params.yaml')
        train_data = load_feature(datapath='data/processed')
        X_train = train_data.iloc[:, :-1]
        Y_train = train_data.iloc[:, -1]
        clf = train_model(X_train, Y_train, n_estimator=n_estimator, learn_rate=learn_rate)
        save_model(clf)
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")

if __name__ == '__main__':
    main()
