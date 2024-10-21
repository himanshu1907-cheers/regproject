import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pickle

# Load the saved model
filename = 'random_forest_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

# Load the dataset (replace with your actual dataset)
df = pd.read_csv('/content/Loan.csv')
df = df.drop(['ApplicationDate', 'LoanApproved'], axis=1)
le = LabelEncoder()
for column in ['EmploymentStatus', 'EducationLevel', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose']:
    df[column] = le.fit_transform(df[column])
y = df['RiskScore']
X = df.drop('RiskScore', axis=1)

# Create a Streamlit app
st.title('Loan Risk Prediction App')

# Sidebar for user input
st.sidebar.header('Input Features')

EmploymentStatus = st.sidebar.selectbox('EmploymentStatus', df['EmploymentStatus'].unique())
EducationLevel = st.sidebar.selectbox('EducationLevel', df['EducationLevel'].unique())
MaritalStatus = st.sidebar.selectbox('MaritalStatus', df['MaritalStatus'].unique())
HomeOwnershipStatus = st.sidebar.selectbox('HomeOwnershipStatus', df['HomeOwnershipStatus'].unique())
LoanPurpose = st.sidebar.selectbox('LoanPurpose', df['LoanPurpose'].unique())
LoanAmount = st.sidebar.number_input('LoanAmount')
CreditScore = st.sidebar.number_input('CreditScore')
LoanTerm = st.sidebar.number_input('LoanTerm')
MonthlyDebt = st.sidebar.number_input('MonthlyDebt')
Income = st.sidebar.number_input('Income')
YearsInCurrentJob = st.sidebar.number_input('YearsInCurrentJob')

# Create a dictionary of the user input
user_input = {
    'EmploymentStatus': EmploymentStatus,
    'EducationLevel': EducationLevel,
    'MaritalStatus': MaritalStatus,
    'HomeOwnershipStatus': HomeOwnershipStatus,
    'LoanPurpose': LoanPurpose,
    'LoanAmount': LoanAmount,
    'CreditScore': CreditScore,
    'LoanTerm': LoanTerm,
    'MonthlyDebt': MonthlyDebt,
    'Income': Income,
    'YearsInCurrentJob': YearsInCurrentJob
}

# Create a dataframe from the user input
input_df = pd.DataFrame([user_input])

# Make prediction using the loaded model
prediction = loaded_model.predict(input_df)

# Display the prediction
st.subheader('Prediction')
st.write(f'The predicted Risk Score is: {prediction[0]:.2f}')

# Display some evaluation metrics if needed
st.subheader('Model Evaluation')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = loaded_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
st.write(f'R-squared: {r2:.2f}')
st.write(f'Mean Squared Error: {mse:.2f}')