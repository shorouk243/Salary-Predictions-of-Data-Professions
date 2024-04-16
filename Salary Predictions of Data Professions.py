#!/usr/bin/env python
# coding: utf-8

# In[390]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# In[391]:


df = pd.read_csv("/Users/shero/Documents/Artificial Intelligence/Machine Learning/Salary Prediction of Data Professions/Salary Prediction of Data Professions.csv")


# In[392]:


df


# # Data Cleaning 

# In[393]:


duplicate = df [df.duplicated()]
duplicate


# In[394]:


df.info()


# In[395]:


df.isnull().all()


# In[396]:


df.fillna(0, inplace=True)


# In[397]:


df['SALARY'] = df['SALARY'].astype(float)
df['PAST EXP'] = df['PAST EXP'].astype(float)


# In[398]:


df.info()


# In[399]:


df.describe()


# # Tasks

# ### Exploratory Data Analysis (EDA):

# In[400]:


plt.figure(figsize=(10, 6))
sns.histplot(df['SALARY'], kde=True)
plt.title('Distribution of Salaries')
plt.xlabel('SALARY')
plt.ylabel('Count')
plt.show()


# In[402]:


plt.figure(figsize=(10, 6))
sns.histplot(df['PAST EXP'], kde=True)
plt.title('Distribution of Past Experience')
plt.xlabel('Past Experience')
plt.ylabel('Count')
plt.show()


# In[403]:


plt.figure(figsize=(10, 6))
sns.histplot(df['RATINGS'], kde=True)
plt.title('Distribution of Ratings')
plt.xlabel('Ratings')
plt.ylabel('Count')
plt.show()


# In[404]:


plt.figure(figsize=(10, 6))
sns.histplot(df['DESIGNATION'], kde=True)
plt.title('Distribution of DESIGNATION')
plt.xlabel('DESIGNATION')
plt.ylabel('Count')
plt.show()


# In[405]:


numeric_features = ['PAST EXP', 'AGE', 'RATINGS']
for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=feature, y='SALARY')
    plt.title(f'{feature.capitalize()} vs. SALARY')
    plt.xlabel(feature.capitalize())
    plt.ylabel('SALARY')
    plt.show()


# In[408]:


categorical_features = ['PAST EXP', 'AGE', 'RATINGS']
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x=feature, y='SALARY')
    plt.title(f'{feature.capitalize()} vs. SALARY')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Salary')
    plt.xticks(rotation=45)
    plt.show()


# ### Feature Engineering:

# In[409]:


df['DOJ'] = pd.to_datetime(df['DOJ'], errors='coerce')
df['CURRENT DATE'] = pd.to_datetime(df['CURRENT DATE'], errors='coerce')

df['Years of Experience'] = (df['CURRENT DATE'] - df['DOJ']).dt.days / 365

df['Years of Experience'] = np.where(df['Years of Experience'] < 0, np.nan, df['Years of Experience'])

df['Years of Experience'] = df['Years of Experience'].round(1)

df['LEAVE EFFICIENCY'] = df['LEAVES USED'] / (df['LEAVES USED'] + df['LEAVES REMAINING'])

age_bins = [20, 30, 40, 50, 60, 100]
age_labels = ['20-30', '30-40', '40-50', '50-60', '60+']
df['AGE GROUP'] = pd.cut(df['AGE'], bins=age_bins, labels=age_labels, right=False)

df_encoded = pd.get_dummies(df, columns=['DESIGNATION', 'SEX', 'UNIT', 'AGE GROUP'], drop_first=True)

df_encoded.drop(['FIRST NAME', 'LAST NAME', 'DOJ', 'CURRENT DATE'], axis=1, inplace=True)

df_encoded.head()


# In[410]:


# df_encoded["Years of Experience"].fillna("No Years of Experience", inplace = True)


# ### Data Preprocessing:

# In[411]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = df_encoded.drop('SALARY', axis=1)
y = df_encoded['SALARY']

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("training data:")
print(X_train.head())
print("\ntesting data:")
print(X_test.head())


# ### Machine Learning Model Development:

# In[412]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define a function to train and evaluate a regression model
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Display evaluation metrics
    print(f"Model: {type(model).__name__}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")
    print("\n")

# Train and evaluate different regression models
linear_reg_model = LinearRegression()
decision_tree_model = DecisionTreeRegressor(random_state=42)
random_forest_model = RandomForestRegressor(random_state=42)
gradient_boosting_model = GradientBoostingRegressor(random_state=42)

# Train and evaluate each model
train_and_evaluate_model(linear_reg_model, X_train_scaled, y_train, X_test_scaled, y_test)
train_and_evaluate_model(decision_tree_model, X_train_scaled, y_train, X_test_scaled, y_test)
train_and_evaluate_model(random_forest_model, X_train_scaled, y_train, X_test_scaled, y_test)
train_and_evaluate_model(gradient_boosting_model, X_train_scaled, y_train, X_test_scaled, y_test)


# ### Model Evaluation:

# In[413]:


# Define a function to display evaluation metrics and identify the best model
def display_evaluation_metrics(models, X_train, y_train, X_test, y_test):
    best_model = None
    best_mae = float('inf')  # Initialize with a large value
    
    for model in models:
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        
        # Display evaluation metrics
        print(f"Model: {type(model).__name__}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"R-squared (R2): {r2:.2f}")
        print("\n")
        
        # Identify the best model based on MAE
        if mae < best_mae:
            best_mae = mae
            best_model = model
    
    # Display the best model
    print(f"Best Model: {type(best_model).__name__} (lowest MAE: {best_mae:.2f})")

# List of regression models
models_to_evaluate = [linear_reg_model, decision_tree_model, random_forest_model, gradient_boosting_model]

# Display evaluation metrics and identify the best model
display_evaluation_metrics(models_to_evaluate, X_train_scaled, y_train, X_test_scaled, y_test)


# ### ML Pipelines and Model Deployment:

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Define the column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), ['AGE', 'LEAVES USED', 'LEAVES REMAINING', 'RATINGS', 'PAST EXP']),
        ('cat', OneHotEncoder(drop='first'), ['DESIGNATION', 'SEX', 'UNIT', 'AGE GROUP'])
    ],
    remainder='passthrough'
)

# Create the machine learning pipeline
model = RandomForestRegressor(random_state=42)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Train the pipeline on the entire dataset
pipeline.fit(X, y)


# In[ ]:


from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        input_data = pd.DataFrame(data)
        input_data = input_data.drop(['FIRST NAME', 'LAST NAME', 'DOJ', 'CURRENT DATE'], axis=1)  # Drop unnecessary columns
        input_data['EXPERIENCE'] = (input_data['CURRENT DATE'] - input_data['DOJ']).dt.days / 365.25

        predictions = pipeline.predict(input_data)

        return jsonify(predictions.tolist())

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000)


# In[ ]:




