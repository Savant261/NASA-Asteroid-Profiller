# Import necessary libraries for data handling, clustering, and saving the model
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load Data
# Load the final, cleaned dataset prepared in the previous step.
df = pd.read_csv('nasa_asteroid_final.csv')

# 2. Select Features (The Physics)
# Choose the features that will be used to group the asteroids into families.
# These orbital parameters describe the shape and tilt of the asteroid's orbit.
features = ['semi_major_axis', 'eccentricity', 'inclination']
X = df[features]

# 3. Scale the Data (Crucial)
# Standardize the features to ensure they are on a similar scale.
# This is important for distance-based algorithms like K-Means.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train the Clustering Model (The "Brain")
# Initialize the K-Means algorithm with 3 clusters, based on the findings from the Elbow Method.
# 'n_init=10' runs the algorithm 10 times with different starting points to find the best result.
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
# Fit the model to the scaled data.
kmeans.fit(X_scaled)

# 5. Assign Labels to the Data
# Create a new column 'cluster_label' in the DataFrame.
# Each asteroid is assigned a label (0, 1, or 2) corresponding to the cluster it belongs to.
df['cluster_label'] = kmeans.labels_

# 6. Save Everything for the Next Steps
# Save the trained K-Means model (the "Brain") and the scaler (the "Translator") using pickle.
# This allows us to use them in the web application without retraining.
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
    
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the DataFrame, now with the cluster labels, to a new CSV file.
# This file will be used in the next stage of the project.
df.to_csv('nasa_asteroid_clustered.csv', index=False)

# Print a success message to the console
print("✅ SUCCESS!")
print("   - Model saved as 'kmeans_model.pkl'")
print("   - Scaler saved as 'scaler.pkl'")
print("   - Data with clusters saved as 'nasa_asteroid_clustered.csv'")
print("   - Ready for Day 3 (Regression)!")