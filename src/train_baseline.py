import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Paths
DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'
MODELS_DIR.mkdir(exist_ok=True)

def load_data():
    """Load and verify training data"""
    csv_path = DATA_DIR / 'student_performance_sample.csv'
    df = pd.read_csv(csv_path)
    
    feature_cols = ['study_hours_per_week', 'attendance_percent', 'previous_grade', 'assignment_score']
    target_col = 'final_grade'
    
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df[target_col]
    
    return X, y, feature_cols

def train_models(X, y):
    """Train multiple models with cross-validation"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    results = {}
    
    # Random Forest
    print("\n[1/2] Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    results['Random Forest'] = {
        'r2': rf_r2,
        'mae': mean_absolute_error(y_test, rf_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred))
    }
    models['rf'] = rf
    print(f"   R² Score: {rf_r2:.4f}")
    
    # Linear Regression (baseline)
    print("[2/2] Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_r2 = r2_score(y_test, lr_pred)
    results['Linear Regression'] = {
        'r2': lr_r2,
        'mae': mean_absolute_error(y_test, lr_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, lr_pred))
    }
    models['lr'] = lr
    print(f"   R² Score: {lr_r2:.4f}")
    
    best_model = 'rf'
    best_r2 = rf_r2
    
    return models, results, scaler, best_model, (X_test, y_test)

def save_models(models, scaler):
    """Save trained models to disk"""
    with open(MODELS_DIR / 'random_forest.pkl', 'wb') as f:
        pickle.dump(models['rf'], f)
    
    with open(MODELS_DIR / 'linear_regression.pkl', 'wb') as f:
        pickle.dump(models['lr'], f)
    
    with open(MODELS_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\n✓ Models saved to models/")

def print_results(results, X_test, y_test):
    """Print training results summary"""
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  R² Score:  {metrics['r2']:.4f}")
        print(f"  MAE:       {metrics['mae']:.4f}")
        print(f"  RMSE:      {metrics['rmse']:.4f}")
    
    best_model = max(results, key=lambda x: results[x]['r2'])
    print(f"\n→ Best Model: {best_model} (R² = {results[best_model]['r2']:.4f})")
    print("="*60)

def main():
    print("Loading data...")
    X, y, feature_cols = load_data()
    print(f"✓ Loaded {len(X)} samples with {len(feature_cols)} features")
    
    print("\nTraining models...")
    models, results, scaler, best_model, (X_test, y_test) = train_models(X, y)
    
    save_models(models, scaler)
    print_results(results, X_test, y_test)
    
    print("\nTraining complete!")

if __name__ == '__main__':
    main()


# Paths
DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'
MODELS_DIR.mkdir(exist_ok=True)

def load_data():
    """Load and verify training data"""
    csv_path = DATA_DIR / 'student_performance_sample.csv'
    df = pd.read_csv(csv_path)
    
    feature_cols = ['hours_studied', 'attendance_rate', 'previous_score', 
                    'sleep_hours', 'extracurricular', 'family_income']
    target_col = 'performance_score'
    
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df[target_col]
    
    return X, y, feature_cols

def train_models(X, y):
    """Train multiple models with cross-validation"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    results = {}
    
    # Random Forest
    print("\n[1/3] Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    results['Random Forest'] = {
        'r2': rf_r2,
        'mae': mean_absolute_error(y_test, rf_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred))
    }
    models['rf'] = rf
    print(f"   R² Score: {rf_r2:.4f}")
    
    # XGBoost
    print("[2/3] Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1, 
        random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_pred = xgb_model.predict(X_test)
    xgb_r2 = r2_score(y_test, xgb_pred)
    results['XGBoost'] = {
        'r2': xgb_r2,
        'mae': mean_absolute_error(y_test, xgb_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred))
    }
    models['xgb'] = xgb_model
    print(f"   R² Score: {xgb_r2:.4f}")
    
    # Linear Regression (baseline)
    print("[3/3] Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_r2 = r2_score(y_test, lr_pred)
    results['Linear Regression'] = {
        'r2': lr_r2,
        'mae': mean_absolute_error(y_test, lr_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, lr_pred))
    }
    models['lr'] = lr
    print(f"   R² Score: {lr_r2:.4f}")
    
    # Save best model
    best_model = 'xgb' if results['XGBoost']['r2'] >= results['Random Forest']['r2'] else 'rf'
    best_r2 = max(results['XGBoost']['r2'], results['Random Forest']['r2'])
    
    return models, results, scaler, best_model, (X_test, y_test)

def save_models(models, scaler):
    """Save trained models to disk"""
    with open(MODELS_DIR / 'random_forest.pkl', 'wb') as f:
        pickle.dump(models['rf'], f)
    
    with open(MODELS_DIR / 'xgboost_model.pkl', 'wb') as f:
        pickle.dump(models['xgb'], f)
    
    with open(MODELS_DIR / 'linear_regression.pkl', 'wb') as f:
        pickle.dump(models['lr'], f)
    
    with open(MODELS_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\n✓ Models saved to models/")

def print_results(results, X_test, y_test):
    """Print training results summary"""
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  R² Score:  {metrics['r2']:.4f}")
        print(f"  MAE:       {metrics['mae']:.4f}")
        print(f"  RMSE:      {metrics['rmse']:.4f}")
    
    best_model = max(results, key=lambda x: results[x]['r2'])
    print(f"\n→ Best Model: {best_model} (R² = {results[best_model]['r2']:.4f})")
    print("="*60)

def main():
    print("Loading data...")
    X, y, feature_cols = load_data()
    print(f"✓ Loaded {len(X)} samples with {len(feature_cols)} features")
    
    print("\nTraining models...")
    models, results, scaler, best_model, (X_test, y_test) = train_models(X, y)
    
    save_models(models, scaler)
    print_results(results, X_test, y_test)
    
    print("\nTraining complete! Run evaluate_models.py for detailed analysis.")

if __name__ == '__main__':
    main()
