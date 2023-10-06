#!/usr/bin/env python
# coding: utf-8

# # Title: Data Cleaning System for Machine Learning Performance Enhancement

# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------

# # 2. Problem Statement
# In the data science pipeline, data cleaning is often treated as a routine task without considering its impact on machine learning model performance. This project takes an approach by prioritizing data quality to enhance model accuracy. It addresses data quality issues, such as missing values and outliers, in the context of predicting global COVID-19 vaccination progress. The aim is to optimize data specifically for machine learning, bridging the gap between data cleaning and predictive modeling. This ensures that the models used for public health decision-making are more accurate and reliable.

# # 3. Business Problem
# 
# <b> Business Problem:</b> The COVID-19 pandemic has highlighted the importance of accurate data analysis and predictive 
# modeling to inform public health decisions. However, the business problem we aim to address is as follows:
#     
# + <b>Suboptimal Decision-Making Due to Poor Data Quality:</b> Public health organizations, governments, and policymakers 
# rely on data-driven insights to make critical decisions regarding vaccine distribution and allocation. Inaccurate or incomplete
# data can lead to suboptimal decisions, resulting in challenges such as inefficient vaccine distribution and resource allocation.
# 
# + <b>Lack of Data Cleaning Optimization:</b> Many organizations lack automated and optimized data cleaning systems tailored to
# specific predictive modeling tasks. As a result, the data preprocessing steps are often performed manually, leading to 
# increased time and potential errors in the data cleaning process.
# 
# + <b> Need for Accurate Predictive Models:</b> Accurate predictive models for COVID-19 vaccination progress are essential for
# resource allocation, vaccine distribution planning, and understanding vaccination coverage trends. The quality of data used for
# model training directly impacts the accuracy of these predictions. 
#     
# By addressing this business problem, our project aims to provide a robust and automated data cleaning system that optimizes
# data quality specifically for predictive modeling related to COVID-19 vaccination progress. This will enable public health 
# organizations and policymakers to make data-driven decisions with greater confidence and improve the efficiency of vaccination
# efforts on a global scale.

# # 4. Methodology

# I begin by collecting the dataset from Kaggle, which contains information about COVID-19 vaccination progress worldwide. The dataset includes various features such as country, date, total vaccinations, people vaccinated, and more. You can access the dataset using the following link: COVID-19 World Vaccination Progress.  

# In[1]:


import pandas as pd

# Load the dataset
data = pd.read_csv('country_vaccinations.csv')


# In this code snippet, I load the COVID-19 vaccination dataset using the pandas library. I read the data from a CSV file named 'country_vaccinations.csv' and display the first few rows to inspect its structure in the following section.In this code snippet, we load the COVID-19 vaccination dataset using the pandas library. We read the data from a CSV file named 'country_vaccinations.csv' and display the first few rows to inspect its structure.

# In[ ]:


## 4.2 2ata Exploration
Before diving into data cleaning, let's explore the dataset to gain insights into its structure and contents. We will perform basic statistical analysis and visualization to understand the data better.


# In[2]:


# Explore the dataset
print(data.head())


# In[3]:


print(data.info())


# In[4]:


print(data.describe())


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns
# Visualize data distributions
plt.figure(figsize=(12, 6))
sns.histplot(data["total_vaccinations"], bins=30, kde=True)
plt.title("Distribution of Total Vaccinations")
plt.xlabel("Total Vaccinations")
plt.ylabel("Frequency")
plt.show()


# ## 4.3	Model Before Data Cleaning

# In this section, to prove the importance of Data Cleaning and the role it plays in model optimization I will first start by build and evaluating a regression model without performing an advanced data cleaning and optimization steps (I am still going to use drop for null values, because without removing it I cannot build the model initially). I use the raw features and target variable, 'daily_vaccinations_per_hundred,' to train the model. The model is a Linear Regression model, and I assess its performance using Mean Squared Error (MSE) and R-squared (R2) metrics.
# 
# I will also  create a new feature 'vaccination_rate' by calculating the percentage of people vaccinated relative to the population (this is feature engineering). My reason for doing it in this step, in order to be able to compare between the models performance and see if there was a real optimization. 
# 

# In[6]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Remove rows with any missing values
data= data.dropna()

# Create a new feature
data['daily_vaccinations_per_hundred'] = data['daily_vaccinations'] / (data['people_vaccinated'] / 100)


# Define features and target variable
X_raw = data[['total_vaccinations', 'people_fully_vaccinated_per_hundred']]
y_raw = data['daily_vaccinations_per_hundred']

# Split the data into training and testing sets without data cleaning
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

# Build a regression model without data cleaning (example: Linear Regression)
model_raw = LinearRegression()
model_raw.fit(X_train_raw, y_train_raw)

# Make predictions on the test set without data cleaning
y_pred_raw = model_raw.predict(X_test_raw)

# Evaluate the model without data cleaning
mse_raw = mean_squared_error(y_test_raw, y_pred_raw)
r2_raw = r2_score(y_test_raw, y_pred_raw)

print("Model Performance With Data Cleaning:")
print(f'Mean Squared Error (Raw): {mse_raw}')
print(f'R-squared (Raw): {r2_raw}')


# ## 4.4	Model Data Cleaning & Feature Engineering

# <b>4.4.1 Outlier Detection and Handling </b>

#      Outliers can adversely affect model performance. I will employ the Isolation Forest algorithm to detect and handle any outliers as shown below.

# In[18]:


import matplotlib.pyplot as plt

# Visualize my outliers 
plt.figure(figsize=(8, 6))
data.boxplot(column='total_vaccinations')
plt.title('Box plot of total_vaccinations')
plt.show()

# remove rows where total_vaccinations is an outlier
data = data[data['total_vaccinations'] < data['total_vaccinations'].quantile(0.99)]


# In[8]:


from sklearn.ensemble import IsolationForest

outlier_detector = IsolationForest(contamination=0.05)
data['outlier'] = outlier_detector.fit_predict(data[['total_vaccinations']])
# Keep only inliers
data = data[data['outlier'] == 1]  


# <b> 4.4.2 Data Encoding and Scaling </b>

# In[9]:


from sklearn.preprocessing import StandardScaler, LabelEncoder

# Encode my categorical variable
label_encoder = LabelEncoder()
data['source_name_encoded'] = label_encoder.fit_transform(data['source_name'])

# my scale numerical variable
scaler = StandardScaler()
data['total_vaccinations_per_hundred_scaled'] = scaler.fit_transform(data[['total_vaccinations_per_hundred']])


# <b> 4.4.3. Data Splitting </b>

# In[10]:


from sklearn.model_selection import train_test_split

# Define my features and target variable
X = data[['source_name_encoded', 'total_vaccinations_per_hundred_scaled']]
y = data['daily_vaccinations_per_hundred']

# Split the data into training and testing sets
X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned = train_test_split(X, y, test_size=0.2, random_state=42)


# <b> 4.4.4 Building Model and Evaluation </b>
# 

# In[19]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Build a regression model with my cleaned data
model_cleaned = LinearRegression()
model_cleaned.fit(X_train_cleaned, y_train_cleaned)

# Make predictions on the test set with the cleaned data
y_pred_cleaned = model_cleaned.predict(X_test_cleaned)

# Evaluate the model with the cleaned data
mse_cleaned = mean_squared_error(y_test_cleaned, y_pred_cleaned)
r2_cleaned = r2_score(y_test_cleaned, y_pred_cleaned)

print("Model Performance With Data Cleaning:")
print(f'Mean Squared Error (cleaned): {mse_cleaned}')
print(f'R-squared (cleaned): {r2_cleaned}')


# --------------------------------------------------------------------------------------------------------------------------------

# # 5. Comparative Analysis
# 
# 1.	Mean Squared Error (MSE): The model with data cleaning has a lower MSE “4.62” compared to the model without data cleaning “6.14”. A lower MSE indicates that the model's predictions are closer to the actual values, suggesting better predictive accuracy in the presence of data cleaning.
# 2.	R-squared (R²): The model with data cleaning also has a higher R-squared value “0.26” compared to the model without data cleaning “0.21”. A higher R-squared indicates that a larger proportion of the variance in the dependent variable is explained by the model. Thus, the model with data cleaning provides a better fit to the data.
# 
# In summary, based on these performance metrics, my model with data cleaning outperforms the model without data cleaning in terms of predictive accuracy and goodness of fit to the data. In other words, data cleaning, which includes handling missing values, outlier detection, and feature engineering, has a positive impact on the model's performance by improving its ability to make accurate predictions and this shows the importance of data cleaning in order to optimize our machines. And this will take me to my next step: 
# 

# # 6. Automated Data Cleaning System

# In[8]:


import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Handling Missing Values
def handle_missing_values(data):
    print("Handling missing values...")
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data)
    return pd.DataFrame(data_imputed, columns=data.columns)

# Step 2: Outlier Detection and Handling
def handle_outliers(data):
    print("Handling outliers...")
    outlier_detector = IsolationForest(contamination=0.05)
    data['outlier'] = outlier_detector.fit_predict(data)
    data = data[data['outlier'] == 1]
    return data

# Step 3: Feature Engineering
def feature_engineering(data):
    print("Performing feature engineering...")
    data['daily_vaccinations_per_hundred'] = data['daily_vaccinations'] / (data['people_vaccinated'] / 100)
    return data

# Step 4: Data Encoding and Scaling
def encode_and_scale(data):
    print("Performing data encoding and scaling...")
    label_encoder = LabelEncoder()
    data['source_name_encoded'] = label_encoder.fit_transform(data['source_name'])
    scaler = StandardScaler()
    data['total_vaccinations_per_hundred_scaled'] = scaler.fit_transform(data[['total_vaccinations_per_hundred']])
    return data

# Step 5: Data Splitting
def split_data(data):
    X = data[['source_name_encoded', 'total_vaccinations_per_hundred_scaled']]
    y = data['daily_vaccinations_per_hundred']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Step 6: Model Building and Evaluation
def build_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print model evaluation results
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    return mse, r2


# In[13]:


# Step 7: Pipeline Integration
def data_cleaning_pipeline(data):
    print("Handling missing values...")
    data = handle_missing_values(data)
    
    print("Handling outliers...")
    data = handle_outliers(data)
    
    print("Performing feature engineering...")
    data = feature_engineering(data)
    
    print("Encoding and scaling data...")
    data = encode_and_scale(data)
    
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(data)
    
    print("Building and evaluating the model...")
    mse, r2 = build_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    return mse, r2


# In[14]:


# Step 9: Automation Script
def automated_data_cleaning_script():
    # Set up logging
    logging.basicConfig(filename='data_cleaning_log.txt', level=logging.INFO)
    
    try:
        # Load raw data
        print("Loading raw data...")
        data = pd.read_csv('country_vaccinations.csv')
        
        # Apply data cleaning pipeline
        print("Applying data cleaning pipeline...")
        mse, r2 = data_cleaning_pipeline(data)
        
        # Log and print performance
        logging.info(f'Mean Squared Error: {mse}')
        logging.info(f'R-squared: {r2}')
        print(f'Mean Squared Error: {mse}')
        print(f'R-squared: {r2}')
    
    except Exception as e:
        # Handle exceptions and log errors
        logging.error(f'Error: {str(e)}')


# In[17]:


# Call the main script function
import logging
automated_data_cleaning_script()


# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------
