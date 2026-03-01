import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import cross_val_score
import numpy as np

def evaluate_all_models():
    """Evaluate all trained models with metrics"""
    
    # Load data
    data_path = Path(__file__).resolve().parents[1] / 'data' / 'student_performance_sample.csv'
    df = pd.read_csv(data_path)
    
    feature_cols = ['study_hours_per_week', 'attendance_percent', 'previous_grade', 'assignment_score']
    X = df[feature_cols]
    y = df['final_grade']
    
    models_dir = Path(__file__).resolve().parents[1] / 'models'
    
    results_summary = []
    
    # Evaluate each model
    for model_file in models_dir.glob('*.pkl'):
        if 'scaler' in model_file.name:
            continue
        
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        try:
            # 5-fold cross-validation
            cv_scores = cross_val_score(
                model, X, y, cv=5, 
                scoring='r2'
            )
            
            results_summary.append({
                'model': model_file.stem,
                'mean_cv_r2': cv_scores.mean(),
                'std_cv_r2': cv_scores.std(),
                'cv_scores': cv_scores
            })
        except:
            print(f"Could not evaluate {model_file.name}")
    
    return results_summary

def generate_report():
    """Generate evaluation report"""
    print("\n" + "="*70)
    print("MODEL EVALUATION REPORT")
    print("="*70)
    
    results = evaluate_all_models()
    
    for result in sorted(results, key=lambda x: x['mean_cv_r2'], reverse=True):
        print(f"\nModel: {result['model']}")
        print(f"  Mean CV R² Score: {result['mean_cv_r2']:.4f} (±{result['std_cv_r2']:.4f})")
        print(f"  Individual Fold Scores: {[f'{s:.3f}' for s in result['cv_scores']]}")
    
    best = max(results, key=lambda x: x['mean_cv_r2'])
    print(f"\n→ Best Model: {best['model']} (Mean R² = {best['mean_cv_r2']:.4f})")
    print("="*70 + "\n")

if __name__ == '__main__':
    generate_report()

    
    models_dir = Path(__file__).resolve().parents[1] / 'models'
    
    results_summary = []
    
    # Evaluate each model
    for model_file in models_dir.glob('*.pkl'):
        if 'scaler' in model_file.name:
            continue
        
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        try:
            # 5-fold cross-validation
            cv_scores = cross_val_score(
                model, X, y, cv=5, 
                scoring='r2'
            )
            
            results_summary.append({
                'model': model_file.stem,
                'mean_cv_r2': cv_scores.mean(),
                'std_cv_r2': cv_scores.std(),
                'cv_scores': cv_scores
            })
        except:
            print(f"Could not evaluate {model_file.name}")
    
    return results_summary

def generate_report():
    """Generate evaluation report"""
    print("\n" + "="*70)
    print("MODEL EVALUATION REPORT")
    print("="*70)
    
    results = evaluate_all_models()
    
    for result in sorted(results, key=lambda x: x['mean_cv_r2'], reverse=True):
        print(f"\nModel: {result['model']}")
        print(f"  Mean CV R² Score: {result['mean_cv_r2']:.4f} (±{result['std_cv_r2']:.4f})")
        print(f"  Individual Fold Scores: {[f'{s:.3f}' for s in result['cv_scores']]}")
    
    best = max(results, key=lambda x: x['mean_cv_r2'])
    print(f"\n→ Best Model: {best['model']} (Mean R² = {best['mean_cv_r2']:.4f})")
    print("="*70 + "\n")

if __name__ == '__main__':
    generate_report()
