# Customer Churn Prediction
## Week 1: Exploratory Data Analysis
### Dataset- Source: Telco Customer Churn Dataset- Size: 7,043 customers, 21 features- Target: Predict customer churn (Yes/No)
### Files- `week1_eda.ipynb` - Exploratory analysis notebook- `customer_data.csv` - Dataset (not included, download separately)
### Key Findings

1: Churn Rate: 26.54% of customers have churned, while 73.46% stayed.

2: Tenure is the strongest predictor: Customers with longer tenure are significantly less likely to churn (strong negative correlation).

3: Monthly Charges impact churn: Higher monthly charges are associated with higher churn probability.

4: Senior Citizens churn more frequently: Senior customers show a higher likelihood of churn compared to non-senior  customers.

5: Data Quality Note: TotalCharges contained 11 missing values, which require cleaning before modeling.
### Next Steps- Week 2: Build machine learning models- Week 3: Model optimization- Week 4: Deploy interactive dashboard
### Setup
```bash
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook week1_eda.ipynb
