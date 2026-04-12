import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title='Customer Churn Predictor', page_icon='📊', layout='wide')

# --- Helper Function: Load Model ---
@st.cache_resource
def load_model():
    # Replace with your actual filename if different
    with open('best_churn_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# List of features the model expects (from your error log)
EXPECTED_COLUMNS = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 
    'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 
    'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 
    'OnlineBackup_No internet service', 'OnlineBackup_Yes', 
    'DeviceProtection_No internet service', 'DeviceProtection_Yes', 
    'TechSupport_No internet service', 'TechSupport_Yes', 
    'StreamingTV_No internet service', 'StreamingTV_Yes', 
    'StreamingMovies_No internet service', 'StreamingMovies_Yes', 
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 
    'PaymentMethod_Mailed check'
]

st.title('📉 Customer Churn Prediction System')

# --- Sidebar / Input Section ---
with st.sidebar:
    st.header('Customer Details')
    gender = st.selectbox('Gender', ['Female', 'Male'])
    senior = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Partner', ['No', 'Yes'])
    dependents = st.selectbox('Dependents', ['No', 'Yes'])
    
    st.divider()
    st.header('Subscription Info')
    tenure = st.slider('Tenure (months)', 0, 72, 12)
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless = st.selectbox('Paperless Billing', ['No', 'Yes'])
    payment = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.number_input('Monthly Charges ($)', 0.0, 200.0, 70.0)
    total_charges = tenure * monthly_charges # Estimated

    st.divider()
    st.header('Services')
    internet = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
    protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
    phone = st.selectbox('Phone Service', ['No', 'Yes'])
    lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])

# --- Prediction Logic ---
if st.button('🔍 Run Churn Analysis', type='primary', use_container_width=True):
    # Prepare data for encoding
    input_dict = {
        'SeniorCitizen': 1 if senior == 'Yes' else 0,
        'tenure': tenure, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
        'gender': gender, 'Partner': partner, 'Dependents': dependents,
        'PhoneService': phone, 'MultipleLines': lines, 'InternetService': internet,
        'OnlineSecurity': security, 'OnlineBackup': backup, 'DeviceProtection': protection,
        'TechSupport': support, 'StreamingTV': tv, 'StreamingMovies': movies,
        'Contract': contract, 'PaperlessBilling': paperless, 'PaymentMethod': payment
    }

    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df)
    
    # Align columns with model expectation
    input_final = input_encoded.reindex(columns=EXPECTED_COLUMNS, fill_value=0)

    # Predict
    prob = model.predict_proba(input_final)[0][1] * 100
    prediction = model.predict(input_final)[0]

    # --- Visualizations ---
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Churn Risk Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Risk (%)", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "green" if prob < 50 else "red"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(0, 255, 0, 0.1)'},
                    {'range': [30, 70], 'color': 'rgba(255, 255, 0, 0.1)'},
                    {'range': [70, 100], 'color': 'rgba(255, 0, 0, 0.1)'}],
            }
        ))
        fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_right:
        st.subheader("Key Risk Factors")
        # Creating a dummy "Impact" chart based on model features
        # In a real app, you'd use model.feature_importances_
        features = ['Tenure', 'Contract', 'Monthly Charges', 'Tech Support']
        impact = [tenure/72, 0.8 if contract == 'Month-to-month' else 0.2, monthly_charges/200, 0.1 if support == 'Yes' else 0.7]
        
        fig_bar = go.Figure(go.Bar(
            x=impact,
            y=features,
            orientation='h',
            marker_color='skyblue'
        ))
        fig_bar.update_layout(height=350, xaxis_title="Relative Impact on Churn")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Final Result Message
    if prediction == 1:
        st.error(f"**Action Required:** High churn probability detected ({prob:.1f}%).")
    else:
        st.success(f"**Safe:** Low churn probability ({prob:.1f}%). High retention likelihood.")