import pandas as pd
import pickle

# Load trained model pipeline
with open('salary_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("=== Salary Prediction ===")

# Take user input and normalize case
try:
    experience = float(input("Enter your years of experience: "))
    job_type = input("Enter your job type (Developer/Manager/Analyst): ").title()
    education = input("Enter your education level (Bachelor/Master): ").title()
except ValueError:
    print("Invalid input!")
    exit()

# Create DataFrame with same column names
input_df = pd.DataFrame({
    'YearsExperience': [experience],
    'JobType': [job_type],
    'EducationLevel': [education]
})

# Predict salary
predicted_salary = model.predict(input_df)

print(f"Predicted Salary: ${predicted_salary[0]:.2f}")
