#  Customer Churn Prediction Project

### Week 1: Exploratory Data Analysis (EDA)

Understanding **why customers leave** is crucial for any business. In this project, I analyzed the **Telco Customer Churn Dataset** to uncover key factors influencing churn and predict which customers are at risk of leaving.  

---

##  Dataset Overview

- **Source:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- **Size:** 7,043 customers  
- **Features:** 21 input variables  
- **Target Variable:** `Churn` (Yes/No)  
- **Objective:** Predict customer churn and identify high-risk segments.  

---

##  Project Files

- `week1_eda.ipynb` – Exploratory Data Analysis notebook  
- `customer_data.csv` – Dataset (download separately)  

---

##  Key Insights from EDA

- **Churn Rate:** 26.54% of customers have churned, while 73.46% remained loyal.  
- **Tenure Matters:** Customers with longer tenure are **less likely to churn**.  
- **High Charges, Higher Risk:** Customers paying higher monthly fees tend to leave more.  
- **Senior Citizens are More Vulnerable:** Seniors show a higher likelihood of churn.  
- **Data Quality Check:** `TotalCharges` column had 11 missing values—cleaning was required before modeling.  

---

##  Week 2: Modeling & Results

### Models Trained

| Model                 | Accuracy |
|----------------------|----------|
| Logistic Regression   | 0.804116   |
| Decision Tree         | 0.794180   |
| Random Forest         | **0.806955** |

**Best Model:** Random Forest delivered the highest accuracy, making it the most reliable predictor for customer churn.  

---

###  Key Takeaways

- **Most Influential Features:**  
  1. `tenure`  
  2. `InternetService_Fiber optic`  
  3. `TotalCharges`  

- **Feature Engineering Impact:**  
  Adding extra features did **not improve performance**, reducing accuracy slightly from 0.8070 to 0.7921. This indicates that the original dataset already captures most predictive patterns for customer churn.  

- **Challenges Overcome:**  
  - Avoided empty DataFrames by ensuring proper dataset loading before feature engineering.  
  - Fixed variable naming conflicts (`df_new`, `df_encoded`) to prevent overwriting.  
  - Resolved `KeyError` issues related to `Churn` and `customerID`.  
  - Handled missing values and datatype issues in `TotalCharges`.  
  - Properly encoded `Churn` before applying one-hot encoding.  

---

##  Next Steps (Week 3)

- Perform **hyperparameter tuning** to improve Random Forest performance.  
- Explore **feature interactions and combinations** to boost model insights.  
- Investigate **class imbalance handling techniques** (e.g., SMOTE, class weighting).  
- Analyze **feature importance** further to refine the predictive model.  

---

## Code Provided by: shaheenaameer2003@gmail.com
## Dataset used: (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
## Dataset name: Telco Customer Churn Dataset

##  Setup Instructions

Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn jupyter
pip install scikit-learn

### Best Model:
- **Optimized XGBoost** with **0.7984%** accuracy
- Saved as: best_churn_model.pkl
### Top 5 Most Important Features:
                 
1. Contract_Two year    
2. InternetService_Fiber optic   
3. Contract_One year   
4. PaymentMethod_Electronic check   
5. InternetService_No    
                       

### Key Learnings:
- Cross-validation provides more reliable estimates than single train/test split
- Hyperparameter tuning can improve accuracy by 2-4%
- XGBoost typically outperforms other algorithms on tabular data
- Feature engineering matters more than model selection
### Next Steps (Week 4):
- Deploy model as interactive web application
- Create user interface for predictions
- Add explanation capabilities
