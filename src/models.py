import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class StudentPerformancePredictor:
    """Inference class for student performance prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = None
        self.model_path = None
    
    def load_model(self, model_path):
        """Load a trained model"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.model_path = model_path
    
    def load_scaler(self, scaler_path):
        """Load feature scaler"""
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
    
    def predict(self, features_dict):
        """Make prediction on new data"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        features = list(features_dict.values())
        features_array = np.array(features).reshape(1, -1)
        
        # Check if model needs scaling (for LR models)
        if self.scaler is not None:
            features_array = self.scaler.transform(features_array)
        
        prediction = self.model.predict(features_array)
        return prediction[0]
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        return metrics, y_pred

def list_available_models():
    """List all trained models"""
    models_dir = Path(__file__).resolve().parents[1] / 'models'
    models = list(models_dir.glob('*.pkl'))
    return {m.stem: m for m in models if 'scaler' not in m.stem}
