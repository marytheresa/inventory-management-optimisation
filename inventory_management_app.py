import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np

# Load the CSV file into a DataFrame
sales = pd.read_csv(r"/Users/chi-chi/Documents/IMO/SalesFINAL12312016.csv")

sales['SalesDate'] = pd.to_datetime(sales['SalesDate'])

sales['WeekdayNum'] = pd.DatetimeIndex(sales['SalesDate']).weekday + 1

def create_time_feature(df):
    df['dayofmonth'] = df['SalesDate'].dt.day
    df['month'] = df['SalesDate'].dt.month
    df['year'] = df['SalesDate'].dt.year
    return df

sales = create_time_feature(sales)

# Access the data in the DataFrame
print(sales.head())
# Load the trained XGBoost model
loaded_model = xgb.Booster()
loaded_model.load_model('/Users/chi-chi/xgboost_model.json')

# Define the features
features = ['Store', 'Brand', 'SalesDollars', 'SalesPrice', 'Volume', 'Classification', 'ExciseTax', 'VendorNo', 'WeekdayNum', 'dayofmonth', 'month', 'year']

# Dictionary to store user inputs
user_input = {}

st.title('Sales Quantity Forecasting')

# Create input fields for each feature
for feature in features:
    if feature in ['Store', 'Brand', 'Classification', 'VendorNo']:
        user_input[feature] = st.selectbox(f'Select {feature}', options=sales[feature].unique())
    else:
        user_input[feature] = float(st.number_input(f'Enter {feature}', value=float(sales[feature].mean())))

# Convert user inputs into a DataFrame
input_data = pd.DataFrame([user_input])

# Extract text features
cats = input_data.select_dtypes(exclude=np.number).columns.tolist()

# Convert to Pandas category
for col in cats:
   input_data[col] = input_data[col].astype('category')

# Create DMatrix for prediction
dmatrix = xgb.DMatrix(input_data, enable_categorical=True)

# Predict
prediction = loaded_model.predict(dmatrix)

st.write('### Forecasted Sales Quantity')
st.write(prediction[0])

if st.button('Show Feature Importance'):
    importance = loaded_model.get_score(importance_type='weight')
    importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
    importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    st.bar_chart(importance_df.set_index('Feature'))
