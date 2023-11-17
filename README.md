
# Customer Churn Prediction - Model Development README
## Objective:
The objective of this code is to develop a predictive model for customer churn using machine learning techniques.

## Tools and Libraries:
Pandas
NumPy
Seaborn
Matplotlib
Scikit-learn
Keras
## Steps:
### 1. Loading the Dataset
The dataset is loaded from Google Drive, titled CustomerChurn_dataset.csv.
### 2. Exploratory Data Analysis (EDA)
Visual analysis and statistical summaries are conducted to understand the data distribution and relationships.
Data preprocessing involves handling categorical values and imputing missing data in the 'TotalCharges' column.
### 3. Feature Engineering
Scaling the data for model training.
Determining feature importance using a RandomForestClassifier to select the most significant features for model training.
### 4. Model Training
Training a RandomForestClassifier to determine feature importance.
Building and training a Multi-Layer Perceptron (MLP) model using Keras.
### 5. Model Evaluation
Evaluating the model's accuracy and performance metrics (accuracy score, AUC score) on the test set.
### 6. Model Deployment Preparation
Saving the trained model, scaler, and label encoder for future deployment.
## Usage:
The provided code should be executed sequentially in an environment that supports Python and the required libraries.
Ensure access to the dataset location and required permissions for saving model-related files.
Model evaluation and deployment-related sections should be adapted based on specific deployment environments.
## Files:
CustomerChurn_dataset.csv: Input dataset for customer churn prediction.
model3.pkl: Saved model file.
scaler3.pkl: Saved scaler file.
label3.pkl: Saved label encoder file.
Customer_churn.ipynb: notebook file.
app2.py: streamlit deployment file
