# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load your dataset
# Example: Replace this with the path to your dataset
df = pd.read_csv(r"C:\Users\hi\Downloads\Disease_symptom_and_patient_profile_dataset.csv")
#
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the feature categories
binary_cols = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
categorical_cols = ['Blood Pressure', 'Cholesterol Level', 'Gender']
numerical_cols = ['Age']

# Define transformers
ordinal_transformer = OrdinalEncoder(categories=[['No', 'Yes']] * len(binary_cols))
onehot_transformer = OneHotEncoder(drop='first', sparse=False)
scaler = StandardScaler()

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('binary', ordinal_transformer, binary_cols),
        ('categorical', onehot_transformer, categorical_cols),
        ('scaler', scaler, numerical_cols)
    ]
)





# Feature selection: X contains the input features, y contains the target (binary label)
X = df.iloc[:,:-1]  # Replace with your features
y = df.iloc[:,-1]  # Replace with your target column

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize logistic regression model
log_reg = LogisticRegression()

# Train the model
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the evaluation results
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

