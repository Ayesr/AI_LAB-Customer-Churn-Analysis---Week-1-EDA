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
        background-color: #f8f9fa;
    }
    .stButton>button {
        border-radius: 10px;
        height: 3.5em;
        font-weight: bold;
        background-color: #007bff;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #007bff !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Function: Load Model ---
@st.cache_resource
def load_model():
    # Ensure 'best_churn_model.pkl' exists in your directory
    with open('best_churn_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Expected features from the training phase
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

# --- Header Section ---
st.title('📉 Customer Churn Prediction Dashboard')
st.markdown("### Strategic Analytics for Customer Retention")
st.divider()

# --- Input Section (Horizontal/Grid Layout) ---
st.subheader("1. Configure Customer Profile")
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

st.markdown("<br>", unsafe_allow_html=True) # Spacer

# --- Prediction Logic ---
if st.button('🔍 RUN CHURN ANALYSIS', use_container_width=True):
    # Data Preparation
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

    # Predict Probabilities
    prob = model.predict_proba(input_final)[0][1] * 100
    prediction = model.predict(input_final)[0]

    st.divider()
    st.subheader("2. Risk Assessment Results")

    # --- Results Visualization ---
    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Determine gauge color dynamically
        if prob < 30:
            bar_color = "#2ECC71"  # Success Green
            status_msg = "LOW RISK"
        elif prob < 70:
            bar_color = "#F1C40F"  # Warning Yellow
            status_msg = "MEDIUM RISK"
        else:
            bar_color = "#E74C3C"  # Danger Red
            status_msg = "HIGH RISK"

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={'suffix': "%", 'font': {'size': 50}},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Assessment: {status_msg}", 'font': {'size': 24, 'color': bar_color}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': bar_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#d3d3d3",
                'steps': [
                    {'range': [0, 30], 'color': "#D5F5E3"},   # Emerald Green
                    {'range': [30, 70], 'color': "#FCF3CF"},  # Sunflower Yellow
                    {'range': [70, 100], 'color': "#FADBD8"}  # Alizarin Red
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': prob
                }
            }
        ))
        fig_gauge.update_layout(height=400, margin=dict(l=30, r=30, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_right:
        st.markdown("### Risk Factor Impact")
        # Example feature impact (In a production app, use SHAP or model.feature_importances_)
        features = ['Tenure Duration', 'Contract Type', 'Monthly Charges', 'Tech Support']
        impact = [
            (72 - tenure) / 72, 
            0.9 if contract == 'Month-to-month' else 0.1, 
            monthly_charges / 200, 
            0.8 if support == 'No' else 0.2
        ]
        
        fig_bar = go.Figure(go.Bar(
            x=impact, y=features, orientation='h',
            marker_color=['#3498DB', '#9B59B6', '#E67E22', '#1ABC9C']
        ))
        fig_bar.update_layout(height=350, xaxis_title="Relative Contribution to Churn", margin=dict(t=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    # Final Summary Message
    if prediction == 1:
        st.error(f"**Critical Alert:** This customer is likely to churn. Recommended action: Offer a long-term contract or loyalty discount.")
    else:
        st.success(f"**Retention Outlook:** This customer is stable. Continue standard engagement protocols.")
