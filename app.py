import streamlit as st
import pandas as pd
import pickle

# Load trained model pipeline
with open('salary_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("💰 Salary Prediction App")
st.write("Predict salary based on experience, job type, and education level.")

# Input fields
experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.5)

job_type = st.selectbox("Job Type", ["Developer", "Manager", "Analyst"])
education = st.selectbox("Education Level", ["Bachelor", "Master"])

# Predict button
if st.button("Predict Salary"):
    # Create DataFrame for model
    input_df = pd.DataFrame({
        'YearsExperience': [experience],
        'JobType': [job_type],
        'EducationLevel': [education]
    })
    
    # Predict salary
    predicted_salary = model.predict(input_df)[0]
    
    # Display result
    st.success(f"Predicted Salary: ${predicted_salary:,.2f}")
