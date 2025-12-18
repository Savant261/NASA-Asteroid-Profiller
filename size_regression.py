# Import necessary libraries for data handling, model training, and evaluation
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the Hybrid Data
# This dataset contains both measured and unmeasured asteroids, now with cluster labels.
df = pd.read_csv('nasa_asteroid_clustered.csv')

# 2. Separate Data for Training and Prediction
# We will train our regression model only on asteroids with known diameters.
train_df = df[df['dataset_type'] == 'measured'].copy()
# The model will then be used to predict the diameters of the unmeasured asteroids.
predict_df = df[df['dataset_type'] == 'unmeasured'].copy()

print(f"🎓 Training on {len(train_df)} measured asteroids...")

# 3. Select Features (X) and Target (y)
# The features are the properties we will use to predict the diameter.
# We use absolute magnitude (a proxy for brightness) and orbital parameters.
# We do not use albedo (reflectivity) because it's unknown for the unmeasured asteroids.
features = ['magnitude_h', 'semi_major_axis', 'eccentricity', 'inclination']
# The target is the value we want to predict: the diameter.
target = 'diameter'

X = train_df[features]
y = train_df[target]

# 4. Split the Training Data for Evaluation
# We set aside 20% of the measured data as a test set to evaluate the model's performance.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Initialize the Regression Model
# We use a RandomForestRegressor, which is an ensemble of decision trees, good for this type of prediction task.
regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# 6. Train the Model
# Fit the model to the training data.
regressor.fit(X_train, y_train)

# 7. Evaluate the Model's Performance
# Make predictions on the test set.
predictions = regressor.predict(X_test)
# Calculate Mean Absolute Error (MAE): On average, how far off are the predictions in km.
mae = mean_absolute_error(y_test, predictions)
# Calculate R-squared (R2) Score: How well the model explains the variance in the data (closer to 1 is better).
r2 = r2_score(y_test, predictions)

print("--- Model Report Card ---")
print(f"✅ Training Complete!")
print(f"   Mean Error: +/- {mae:.3f} km")
print(f"   Accuracy Score (R2): {r2:.3f} (Closer to 1.0 is better)")

# 8. Predict Diameters for Unmeasured Asteroids
# Now, use the trained model to predict the diameters of the asteroids in the 'unmeasured' dataset.
X_unmeasured = predict_df[features]
predicted_sizes = regressor.predict(X_unmeasured)

# 9. Update the Main DataFrame
# Fill in the missing 'diameter' values in the main dataframe with our new predictions.
df.loc[df['dataset_type'] == 'unmeasured', 'diameter'] = predicted_sizes

# 10. Save the Model and the Final Dataset
# Save the trained regressor model for later use in the web app.
with open('diameter_regressor.pkl', 'wb') as f:
    pickle.dump(regressor, f)

# Save the complete dataframe, now with all diameters filled in, to a new CSV.
# This is the final dataset that will be used by the classification model and the web app.
df.to_csv('nasa_asteroid_predicted.csv', index=False)
print("---")
print("💾 Saved 'diameter_regressor.pkl' (The Sizer Brain)")
print("💾 Saved 'nasa_asteroid_predicted.csv' (Full dataset with predicted sizes)")