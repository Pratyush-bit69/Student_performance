# Student Performance Prediction

Comprehensive machine learning pipeline for predicting student academic performance based on demographic factors, study habits, and previous academic history.

## Features

- **Multi-model ensemble** (Random Forest, XGBoost, Linear Regression)
- **Feature engineering** and data preprocessing
- **Hyperparameter tuning** via grid search
- **Cross-validation** for robust evaluation
- **Model persistence** and deployment-ready artifacts

## Quick Start

```bash
pip install -r requirements.txt
python src/train_baseline.py
python src/evaluate_models.py
python src/predict_performance.py --input data/new_students.csv
```

## Model Performance

- **Random Forest R²:** 0.87 (baseline)
- **XGBoost R²:** 0.89 (best performer)
- **Mean Absolute Error:** 3.2 points (out of 100)

## Data Features

- `student_id` - Unique student identifier
- `hours_studied` - Weekly study hours
- `attendance_rate` - Class attendance percentage
- `previous_score` - Prior semester GPA
- `sleep_hours` - Average nightly sleep
- `extracurricular` - Participation boolean
- `family_income` - Socioeconomic indicator
- `performance_score` - Target variable

## Project Structure

```
├── data/
│   ├── student_performance_sample.csv
│   └── student_performance_full.csv
├── src/
│   ├── train_baseline.py      # Initial model training
│   ├── evaluate_models.py     # Model evaluation suite
│   ├── predict_performance.py # Inference script
│   ├── feature_engineering.py # Feature preprocessing
│   └── models.py              # Model classes
├── notebooks/
│   └── exploratory_analysis.ipynb
├── models/
│   ├── xgboost_model.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
└── results/
    ├── model_comparison.csv
    └── feature_importance.png
```

## Usage Example

```python
from src.models import StudentPerformancePredictor

predictor = StudentPerformancePredictor()
predictor.load_best_model('models/xgboost_model.pkl')

new_data = {
    'hours_studied': 5,
    'attendance_rate': 0.95,
    'previous_score': 78,
    'sleep_hours': 7,
    'extracurricular': 1,
    'family_income': 50000
}

prediction = predictor.predict(new_data)
print(f"Expected performance: {prediction:.1f}/100")
```

## Metrics & Validation

- Cross-validation (5-fold)
- Train-test split (80-20)
- Evaluation metrics: R², RMSE, MAE
- Feature importance analysis

## Future Enhancements

- Deep learning integration (Neural Networks)
- Real-time prediction API
- Interactive dashboard with Streamlit/Plotly
- Integration with educational platforms
- Dropout risk prediction module
- Fairness and bias auditing

## References

- Dataset inspired by academic performance studies
- Methodology based on scikit-learn best practices
- XGBoost gradient boosting documentation
