# Import necessary libraries for data handling, plotting, and machine learning
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load the clean, combined dataset
df = pd.read_csv('nasa_asteroid_final.csv')

# 2. Select Features for Clustering
# We choose orbital parameters that define the shape and orientation of the asteroid's path.
features = ['semi_major_axis', 'eccentricity', 'inclination']
X = df[features]

# 3. Scale the Data
# This is a crucial step in clustering. It standardizes the features to have a mean of 0 and a standard deviation of 1.
# This prevents features with larger scales (like semi_major_axis) from dominating the clustering algorithm.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. The Elbow Method
# This technique is used to find the optimal number of clusters (k) for the K-Means algorithm.
# We will run K-Means for a range of k values and plot the inertia (within-cluster sum of squares).
inertia = []
k_range = range(1, 11)

print("🚀 Running Elbow Method to find optimal clusters...")

# Loop through each number of clusters from 1 to 10
for k in k_range:
    # Initialize and fit the K-Means model
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    # Append the inertia (error) to our list
    inertia.append(kmeans.inertia_)
    print(f"   - Tested k={k}, Inertia={kmeans.inertia_:.2f}")

# 5. Plot the Results
# The 'elbow' in the plot indicates the optimal k, where adding more clusters doesn't significantly reduce the inertia.
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.title('The Elbow Method: Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Error)')
plt.xticks(k_range)
plt.grid(True)
plt.show()