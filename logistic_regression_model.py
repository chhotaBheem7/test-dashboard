import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import main

df = main.df

# Separate features (X) and target (y)
feature_names = ['Glucose', 'BMI', 'Age']
X = df[feature_names].values  # All features included in the model
y = df.iloc[:, -1].values   # The last feature (target)

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # 80/20 split
)

# Feature Scaling (Important for Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit and transform training data
X_test = scaler.transform(X_test)      # Transform test data using the same scaler

# Test whether the pickle exists.
if os.path.isfile("logistic_regression.mdl"):
    # If the pickle exists, open it.
    infile = open("logistic_regression.mdl", 'rb')
    # Load the pickle.
    model = pickle.load(infile)
    # Close the pickle file.
    infile.close()
    # Output a message to state that the pickle was
    # successfully loaded.
    print("Loaded pickle")

else:
    # Train the Logistic Regression model
    model = LogisticRegression(max_iter=1000, solver='liblinear', C=10.0)
    model.fit(X_train, y_train)
    # Open a file to save the pickle.
    outfile = open("logistic_regression.mdl", "wb")
    # Store the model in the file.
    pickle.dump(model, outfile)
    # Close the pickle file.
    outfile.close()


# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# Classification Report (precision, recall, F1-score)
class_rep = classification_report(y_test, y_pred)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

class_names = np.unique(y_test)  # Or specify manually

df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

# Example of making a prediction on new data (after scaling):
# new_data = np.array([[feature1, feature2, feature3, feature4, feature5]]) # Replace with your values
# new_data_scaled = scaler.transform(new_data) # Scale the new data
# prediction = model.predict(new_data_scaled)
# print("\nPrediction for new data:", prediction)
