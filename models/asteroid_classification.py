# Import necessary libraries for data manipulation, model training, and evaluation
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load the Data
# This dataset now includes the predicted diameters for previously unmeasured asteroids.
df = pd.read_csv('nasa_asteroid_predicted.csv')

# 2. Prepare Training Data
# We will train the classification model only on the data that had original, measured diameters from NASA.
# This ensures the model learns from ground-truth data.
train_df = df[df['dataset_type'] == 'measured'].copy()

# 3. Feature and Target Selection
# Define the features (X) that will be used to predict the target (y).
# Features include orbital parameters, the newly predicted diameter, and the cluster label from the previous step.
features = ['moid_au', 'diameter', 'eccentricity', 'semi_major_axis', 'inclination', 'cluster_label']
# The target is the 'pha_flag', which indicates if an asteroid is a Potentially Hazardous Asteroid.
target = 'pha_flag'

# Separate features and target variable
X = train_df[features]
y = train_df[target]

# 4. Split the Data
# Divide the data into training and testing sets (80% for training, 20% for testing).
# This allows us to evaluate the model's performance on unseen data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Initialize the Model
# We use a RandomForestClassifier, an ensemble model that uses multiple decision trees to make a prediction.
# 'n_estimators=100' means it will build 100 decision trees.
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 6. Train the Model
# Fit the classifier to the training data.
print("🛡️ Training the Hazard Detection System...")
clf.fit(X_train, y_train)

# 7. Make Predictions
# Use the trained model to make predictions on the test set.
y_pred = clf.predict(X_test)

# 8. Evaluate the Model
# Calculate the accuracy of the model.
accuracy = accuracy_score(y_test, y_pred)
print("--- 🛡️ SECURITY REPORT 🛡️ ---")
print(f"✅ Accuracy: {accuracy:.2%}")
print("\n🔍 Detailed Scan Report:")
# Print a detailed classification report including precision, recall, and f1-score.
print(classification_report(y_test, y_pred))

# 9. Visualize the Confusion Matrix
# A confusion matrix shows the number of correct and incorrect predictions for each class.
# This helps to understand where the model is making mistakes.
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Reds')
plt.title('Hazard Detection Matrix')
plt.xlabel('AI Prediction')
plt.ylabel('Actual NASA Label')
plt.show()

# 10. Save the Model
# Save the trained classifier to a file using pickle.
# This allows us to load and use the model later without retraining.
with open('hazard_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("💾 Saved 'hazard_model.pkl' (The Guard Brain)")