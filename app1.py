import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title='Churn Predictor Pro', page_icon='📊', layout='wide')

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        border-radius: 20px;
        height: 3em;
        font-weight: bold;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open('best_churn_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

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

# --- Header ---
st.title('📉 Customer Churn Prediction Dashboard')
st.markdown("Enter customer details across the categories below to analyze retention risk.")
st.divider()

# --- Horizontal Input Layout ---
# We use Tabs to group features and Columns within tabs to spread them out
tab1, tab2, tab3 = st.tabs(["👤 Personal Info", "📑 Subscription Details", "🌐 Services & Support"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        gender = st.selectbox('Gender', ['Female', 'Male'])
    with col2:
        senior = st.selectbox('Senior Citizen', ['No', 'Yes'])
    with col3:
        partner = st.selectbox('Partner', ['No', 'Yes'])
    with col4:
        dependents = st.selectbox('Dependents', ['No', 'Yes'])

with tab2:
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        tenure = st.slider('Tenure (months)', 0, 72, 12)
    with col2:
        contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
    with col3:
        paperless = st.selectbox('Paperless Billing', ['No', 'Yes'])
    
    col4, col5 = st.columns(2)
    with col4:
        payment = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    with col5:
        monthly_charges = st.number_input('Monthly Charges ($)', 0.0, 200.0, 70.0)
    
    total_charges = tenure * monthly_charges

with tab3:
    # Creating a 3x3 grid for services
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        internet = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
        phone = st.selectbox('Phone Service', ['No', 'Yes'])
    with s_col2:
        backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
        protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
        lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
    with s_col3:
        support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
        tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
        movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])

st.divider()

# --- Prediction Logic ---
if st.button('🔍 Run Churn Analysis', type='primary', use_container_width=True):
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
    input_final = input_encoded.reindex(columns=EXPECTED_COLUMNS, fill_value=0)

    # Predict
    prob = model.predict_proba(input_final)[0][1] * 100
    prediction = model.predict(input_final)[0]

    # --- Results Display ---
    st.subheader("Analysis Results")
    
    # Summary Metrics
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        if prediction == 1:
            st.error(f"### High Risk: {prob:.1f}% Churn Probability")
        else:
            st.success(f"### Low Risk: {prob:.1f}% Churn Probability")

    # Visualizations
    col_left, col_right = st.columns([1, 1])

    with col_left:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Probability Gauge", 'font': {'size': 18}},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#EF553B" if prob > 50 else "#00CC96"},
                'steps': [
                    {'range': [0, 30], 'color': "#e8f5e9"},
                    {'range': [30, 70], 'color': "#fffde7"},
                    {'range': [70, 100], 'color': "#ffebee"}]
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=30, r=30, t=50, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_right:
        features = ['Tenure', 'Contract', 'Charges', 'Support']
        impact = [tenure/72, 0.8 if contract == 'Month-to-month' else 0.2, monthly_charges/200, 0.1 if support == 'Yes' else 0.7]
        
        fig_bar = go.Figure(go.Bar(
            x=impact, y=features, orientation='h',
            marker_color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
        ))
        fig_bar.update_layout(height=300, title="Top Risk Contributors", margin=dict(t=50, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)
