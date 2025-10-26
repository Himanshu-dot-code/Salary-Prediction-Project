import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load dataset
data = pd.read_csv('data\salary_data.csv')

# Features and target
X = data[['YearsExperience', 'JobType', 'EducationLevel']]
y = data['Salary']

# Preprocessing: encode categorical features
categorical_features = ['JobType', 'EducationLevel']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), categorical_features)],
    remainder='passthrough'  # keep numerical features as-is
)

# Create a pipeline with preprocessing + Linear Regression
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model_pipeline.fit(X_train, y_train)

# Save the trained pipeline
with open('salary_model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

print("Model trained and saved as salary_model.pkl")
