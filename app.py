import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model (ensure it was trained with 20 features)
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

# App title
st.title("Credit Risk Prediction")
st.write("Enter the details below to predict if the person is a good or bad credit risk.")

# Input fields for user data with proper types and descriptions
age = st.number_input(
    "Enter age:", min_value=18, max_value=100, value=30,
    help="The age of the applicant. Older age might imply more financial stability but could also indicate reduced earning potential."
)
income = st.number_input(
    "Enter monthly income (in $):", min_value=0, value=3000,
    help="The applicant's monthly income. Higher income often correlates with better creditworthiness."
)
loan_amount = st.number_input(
    "Enter loan amount (in $):", min_value=0, value=10000,
    help="The total loan amount the applicant is requesting."
)
loan_duration = st.number_input(
    "Enter loan duration (in months):", min_value=1, value=12,
    help="The length of time (in months) for which the loan is requested."
)
credit_history = st.selectbox(
    "Credit history:", ["Good", "Bad"],
    help="The applicant's credit history, indicating past repayment behavior. 'Good' often reduces risk."
)
employment_status = st.selectbox(
    "Employment Status:", ["Employed", "Unemployed", "Self-employed", "Retired"],
    help="The applicant's employment status, which is a key factor in assessing repayment ability."
)
home_ownership = st.selectbox(
    "Home Ownership Status:", ["Owned", "Rented", "Mortgaged", "Other"],
    help="The applicant's home ownership status, which can indicate asset stability."
)
debt_to_income_ratio = st.number_input(
    "Debt-to-Income Ratio (in %):", min_value=0.0, max_value=100.0, value=30.0,
    help="The percentage of monthly income used for debt repayment. Lower values are better."
)
num_credit_cards = st.number_input(
    "Number of Credit Cards:", min_value=0, value=2,
    help="The number of credit cards the applicant holds. Excessive cards may indicate financial stress."
)
outstanding_debt = st.number_input(
    "Outstanding Debt (in $):", min_value=0, value=5000,
    help="The total unpaid debt of the applicant."
)
loan_purpose = st.selectbox(
    "Purpose of Loan:", ["Personal", "Business", "Education", "Home Improvement", "Medical"],
    help="The reason for requesting the loan. Certain purposes might be riskier than others."
)
marital_status = st.selectbox(
    "Marital Status:", ["Single", "Married", "Divorced", "Widowed"],
    help="The applicant's marital status, which can affect financial obligations and stability."
)
education_level = st.selectbox(
    "Education Level:", ["High School", "Bachelor's", "Master's", "Doctorate", "Other"],
    help="The highest level of education completed by the applicant. Higher education levels may imply better earning potential."
)
financial_dependents = st.number_input(
    "Number of Financial Dependents:", min_value=0, value=1,
    help="The number of people financially dependent on the applicant. Higher values may increase financial strain."
)

# Additional features (total of 20)
employment_duration = st.number_input(
    "Employment Duration (in years):", min_value=0, value=5,
    help="The number of years the applicant has been employed. Longer duration often indicates job stability."
)
num_previous_loans = st.number_input(
    "Number of Previous Loans Taken:", min_value=0, value=1,
    help="The number of loans the applicant has previously taken. More loans may imply financial burden."
)
existing_credit_score = st.number_input(
    "Existing Credit Score:", min_value=0, max_value=850, value=650,
    help="The applicant's existing credit score. Higher scores usually indicate better creditworthiness."
)
location_type = st.selectbox(
    "Location Type:", ["Urban", "Suburban", "Rural"],
    help="The location of the applicant's residence. Urban areas often have more financial opportunities."
)
loan_default_history = st.selectbox(
    "Loan Default History:", ["None", "1-2 Defaults", "3+ Defaults"],
    help="The applicant's previous loan default history. More defaults indicate higher credit risk."
)
previous_default_count = st.number_input(
    "Previous Default Count:", min_value=0, value=0,
    help="The number of defaults the applicant has previously had. Higher counts indicate higher risk."
)

# Encode the text features (like 'credit_history', 'employment_status', etc.) into numeric values
encoder = LabelEncoder()

# Encoding categorical features
credit_history_encoded = 1 if credit_history == "Good" else 0
employment_encoded = {
    "Employed": 1, "Unemployed": 0, "Self-employed": 2, "Retired": 3
}[employment_status]
home_ownership_encoded = {
    "Owned": 1, "Rented": 0, "Mortgaged": 2, "Other": 3
}[home_ownership]
loan_purpose_encoded = {
    "Personal": 0, "Business": 1, "Education": 2, "Home Improvement": 3, "Medical": 4
}[loan_purpose]
marital_status_encoded = {
    "Single": 0, "Married": 1, "Divorced": 2, "Widowed": 3
}[marital_status]
education_level_encoded = {
    "High School": 0, "Bachelor's": 1, "Master's": 2, "Doctorate": 3, "Other": 4
}[education_level]
location_type_encoded = {
    "Urban": 0, "Suburban": 1, "Rural": 2
}[location_type]
loan_default_history_encoded = {
    "None": 0, "1-2 Defaults": 1, "3+ Defaults": 2
}[loan_default_history]

# Prepare input data with 20 features
input_data = np.array([
    age, income, loan_amount, loan_duration, credit_history_encoded, employment_encoded,
    home_ownership_encoded, debt_to_income_ratio, num_credit_cards, outstanding_debt,
    loan_purpose_encoded, marital_status_encoded, education_level_encoded, financial_dependents,
    employment_duration, num_previous_loans, existing_credit_score, location_type_encoded,
    loan_default_history_encoded, previous_default_count
]).reshape(1, -1)

# Predict button
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("Congratulations! Your credit risk is predicted to be GOOD. ✅")
        else:
            st.error("Bad luck! Your credit risk is predicted to be BAD. ❌")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
