# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset (replace 'dataset.csv' with your dataset path)
df = pd.read_csv('dataset.csv')

# Assuming 'Loan_Status' is the target variable column name
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Splitting dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluating model performance
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Saving the model to a pickle file
with open('loan_prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)
