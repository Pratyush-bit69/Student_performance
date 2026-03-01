import argparse
import json
import pickle
from pathlib import Path
from models import StudentPerformancePredictor

def predict_single(student_data):
    """Predict performance for a single student"""
    
    models_dir = Path(__file__).resolve().parents[1] / 'models'
    
    # Load best model (Random Forest)
    predictor = StudentPerformancePredictor()
    predictor.load_model(str(models_dir / 'random_forest.pkl'))
    
    prediction = predictor.predict(student_data)
    
    return {
        'predicted_score': round(prediction, 1),
        'percentile': 'High' if prediction >= 80 else 'Medium' if prediction >= 60 else 'Low',
        'recommendation': get_recommendation(prediction)
    }

def get_recommendation(score):
    """Generate recommendation based on predicted score"""
    if score >= 85:
        return "Excellent performance expected. Consider advanced courses."
    elif score >= 75:
        return "Good performance expected. Continue current study patterns."
    elif score >= 60:
        return "Satisfactory performance. Increase study hours and attendance."
    else:
        return "Support needed. Consider tutoring and increase engagement."

def main():
    parser = argparse.ArgumentParser(description='Predict student performance')
    parser.add_argument('--study_hours_per_week', type=float, default=10)
    parser.add_argument('--attendance_percent', type=float, default=85)
    parser.add_argument('--previous_grade', type=float, default=75)
    parser.add_argument('--assignment_score', type=float, default=75)
    
    args = parser.parse_args()
    
    student_data = {
        'study_hours_per_week': args.study_hours_per_week,
        'attendance_percent': args.attendance_percent,
        'previous_grade': args.previous_grade,
        'assignment_score': args.assignment_score
    }
    
    print("\n" + "="*50)
    print("STUDENT PERFORMANCE PREDICTION")
    print("="*50)
    print("\nInput Profile:")
    for key, value in student_data.items():
        print(f"  {key}: {value}")
    
    result = predict_single(student_data)
    
    print("\nPrediction Result:")
    print(f"  Predicted Score: {result['predicted_score']}/100")
    print(f"  Performance Level: {result['percentile']}")
    print(f"  Recommendation: {result['recommendation']}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
