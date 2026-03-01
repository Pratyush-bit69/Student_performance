import pandas as pd
from pathlib import Path

# Generate and save sample data
data = {
    'hours_studied': [3, 5, 2, 6, 4, 5, 3, 7, 2, 6, 4, 5, 3, 6, 5],
    'attendance_rate': [0.85, 0.95, 0.70, 0.90, 0.80, 0.92, 0.75, 0.98, 0.65, 0.88, 0.82, 0.91, 0.78, 0.93, 0.87],
    'previous_score': [72, 85, 65, 88, 75, 82, 70, 90, 60, 79, 76, 84, 68, 86, 78],
    'sleep_hours': [6, 7, 5, 8, 6, 7.5, 5.5, 8, 4, 7, 6.5, 7, 5.5, 8, 7],
    'extracurricular': [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
    'family_income': [40000, 80000, 35000, 95000, 55000, 75000, 45000, 120000, 30000, 70000, 60000, 85000, 50000, 100000, 65000],
    'performance_score': [72, 88, 64, 91, 76, 85, 71, 94, 62, 82, 78, 87, 70, 89, 80]
}

df = pd.DataFrame(data)
data_dir = Path(__file__).resolve().parents[2] / 'data'
data_dir.mkdir(exist_ok=True)

csv_path = data_dir / 'student_performance_sample.csv'
df.to_csv(csv_path, index=False)
print(f"✓ Sample dataset created: {csv_path}")
print(f"  Shape: {df.shape}")
print(f"  Features: {list(df.columns[:-1])}")
