import numpy as np
import pandas as pd
import os
import pickle
import json
import logging

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger('Model Evaluation')
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

def load_model(url: str):
    try:
        with open(url, 'rb') as file:
            clf = pickle.load(file)
        return clf
    except FileNotFoundError:
        logger.error(f"Error: The file {url} was not found.")
        raise
    except pickle.UnpicklingError:
        logger.error("Error: The file could not be unpickled. It may not be a valid pickle file.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the model: {e}")
        raise

def load_data(datapath: str) -> pd.DataFrame:
    try:
        test_data = pd.read_csv(os.path.join(datapath, 'test_bow.csv'))
        return test_data
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
        logger.error(f"An unexpected error occurred while loading data: {e}")
        raise

def evaluate_model(clf, X_test: pd.DataFrame, Y_test: pd.DataFrame) -> dict:
    try:
        Y_pred = clf.predict(X_test)
        Y_pred_proba = clf.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred)
        recall = recall_score(Y_test, Y_pred)
        auc = roc_auc_score(Y_test, Y_pred_proba)
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        return metrics_dict
    except AttributeError as e:
        logger.error(f"Error: Model does not have the required methods. Details: {e}")
        raise
    except ValueError as e:
        logger.error(f"Error: Value error in predictions or metrics calculation. Details: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while evaluating the model: {e}")
        raise

def save_metrics_dict(metrics: dict, file_path: str) -> None:
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
    except IOError as e:
        logger.error(f"Error: An I/O error occurred while saving the metrics: {e}")
        raise
    except TypeError as e:
        logger.error(f"Error: Type error in metrics dictionary. Details: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving metrics: {e}")
        raise

def main():
    try:
        clf = load_model('models/model.pkl')
        logger.debug('Saved model loaded successfully')
        test_data = load_data(datapath='data/processed')
        logger.debug('Test Data loaded successfully')
        X_test = test_data.iloc[:, :-1]
        Y_test = test_data.iloc[:, -1]
        metrics_dict = evaluate_model(clf, X_test, Y_test)
        logger.debug('Model Evaluation dictionary created successfully')
        save_metrics_dict(metrics_dict, 'reports/metrics.json')
        logger.debug('JSON dump of model evaluation dictionary done')
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")

if __name__ == '__main__':
    main()
