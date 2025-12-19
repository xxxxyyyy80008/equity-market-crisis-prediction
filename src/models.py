import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import joblib
import os
from typing import Dict, Any, Tuple

class CrisisPredictionModel:
    """
    A wrapper class for Gradient Boosted Decision Trees (GBDT) specifically 
    tuned for financial crisis regime prediction.
    """
    
    def __init__(self, model_params: Dict[str, Any] = None):
        """
        Initialize the model with specific hyperparameters.
        
        Args:
            model_params: Dictionary of XGBoost hyperparameters.
        """
        self.default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'n_jobs': -1,
            'random_state': 42
        }
        
        if model_params:
            self.default_params.update(model_params)
            
        self.model = xgb.XGBClassifier(**self.default_params)
        self.is_trained = False

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              eval_set: list = None, early_stopping_rounds: int = 10):
        """
        Train the GBDT model.
        
        Args:
            X_train: Feature matrix.
            y_train: Target vector (Crisis regimes).
            eval_set: List of (X, y) pairs for validation.
            early_stopping_rounds: Rounds to wait for improvement before stopping.
        """
        print(f"Training GBDT model with params: {self.default_params}")
        
        self.model.fit(
            X_train, 
            y_train, 
            eval_set=eval_set, 
            verbose=False
        )
        self.is_trained = True
        print("Training completed.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict crisis regimes.
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of crisis regimes.
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model performance using financial classification metrics.
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
            
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)
        
        metrics = {
            "roc_auc": roc_auc_score(y_test, y_prob),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "report": classification_report(y_test, y_pred, output_dict=True)
        }
        
        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Extract feature importance to understand drivers of crisis regimes.
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
            
        importance = pd.DataFrame({
            'Feature': self.model.feature_names_in_,
            'Importance': self.model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        return importance

    def save(self, filepath: str = 'models/gbdt_crisis_model.joblib'):
        """
        Save the trained model to disk.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str = 'models/gbdt_crisis_model.joblib'):
        """
        Load a trained model from disk.
        """
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            self.is_trained = True
            print(f"Model loaded from {filepath}")
        else:
            raise FileNotFoundError(f"No model found at {filepath}")

if __name__ == "__main__":
    # Example usage for testing the module
    print("Initializing CrisisPredictionModel...")
    model = CrisisPredictionModel()
    print("Model module ready.")