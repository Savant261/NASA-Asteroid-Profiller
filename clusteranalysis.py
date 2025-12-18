# Import the pandas library for data manipulation and analysis
import pandas as pd

# 1. Load the clustered data
# This CSV file contains the original asteroid data along with the 'cluster_label' assigned by the K-Means algorithm.
df = pd.read_csv('nasa_asteroid_clustered.csv')

# 2. Analyze the Clusters
# Group the DataFrame by the 'cluster_label' and calculate the mean (average) of the key orbital features for each cluster.
# This helps in understanding the defining characteristics of each asteroid family identified by the model.
cluster_summary = df.groupby('cluster_label')[['semi_major_axis', 'eccentricity', 'inclination']].mean()

# 3. Count Members in Each Cluster
# Add a 'count' column to the summary, showing how many asteroids belong to each cluster.
cluster_summary['count'] = df['cluster_label'].value_counts()

# 4. Display the Report
# Print the summary table to the console.
print("📊 Cluster Analysis Report:")
print(cluster_summary)