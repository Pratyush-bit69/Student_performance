import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

DATA = Path(__file__).resolve().parents[1] / 'data' / 'student_performance_sample.csv'
df = pd.read_csv(DATA)
X = df[['study_hours_per_week', 'attendance_percent', 'previous_grade', 'assignment_score']]
y = df['final_grade']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print('MAE:', round(mean_absolute_error(y_test, preds), 3))
