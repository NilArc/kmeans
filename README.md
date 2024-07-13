# K-Means Clustering

## Data Used

**Dataset**: Mall_Customers.csv

## Data Information

The dataset contains customer data with the following features:
- **CustomerID**: Unique identifier for each customer.
- **Gender**: Gender of the customer.
- **Age**: Age of the customer.
- **Annual Income (k$)**: Annual income of the customer.
- **Spending Score (1-100)**: Score assigned by the mall based on customer behavior and spending nature.

The objective is to group customers based on their annual income and spending score.

## Problem Statement

Given the dataset, how can we cluster the data points using K-means? The task involves implementing the K-means algorithm to cluster data points based on their distances, determining centroids, and calculating Euclidean distances. The goal is to find the optimal value of K to achieve better clustering results.

## Method

The K-means clustering algorithm will be used to cluster the data based on centroids and Euclidean distances. Initially, K centroids are chosen at random. These centroids represent the center of clusters. The algorithm iteratively refines the centroids to minimize intra-cluster distance and maximize inter-cluster distance.

## Algorithm

The K-means algorithm will be implemented using Python libraries such as NumPy, Pandas, and Matplotlib. The steps are as follows:

1. **Random Initialization**: Choose K random points as centroids.
2. **Assign Points to Clusters**: Assign each data point to the nearest centroid based on Euclidean distance.
3. **Recalculate Centroids**: Calculate the new centroids by averaging the points in each cluster.
4. **Iterate**: Repeat steps 2 and 3 until convergence (no change in centroids) or until a maximum number of iterations is reached.

## Solution

1. **Randomly Select Initial Centroids**: Randomly choose K data points as initial centroids.
2. **Assign Data Points to Nearest Centroid**: Compute the Euclidean distance from each data point to all centroids and assign each point to the nearest centroid.
3. **Recalculate Centroids**: For each cluster, compute the mean of the data points to get the new centroid.
4. **Repeat**: Continue the process of reassigning points and recalculating centroids until the centroids do not change or the maximum number of iterations is reached.

### Steps in Implementation

1. **Load Data**: Load the dataset using Pandas.
2. **Preprocess Data**: Select relevant features (annual income and spending score) and scale the data if necessary.
3. **Initialize K-Means**: Use the KMeans class from scikit-learn to initialize and fit the model.
4. **Determine Optimal K**: Use the elbow method to find the optimal value of K.
5. **Cluster Data**: Fit the K-means model with the optimal K and predict cluster labels for the data points.
6. **Evaluate and Visualize**: Evaluate the model using metrics like inertia and visualize the clusters.

## Evaluation

The K-means algorithm will be evaluated based on the sum of squared distances (inertia) between data points and their respective centroids. The elbow method will be used to determine the optimal number of clusters (K). The consistency and interpretability of the clusters will also be considered.

## Data Preparation

Data preparation includes handling missing values, outliers, and scaling the features. Only numerical features will be used for clustering. The data should be clean and free of noise to ensure accurate clustering.

## Results & Discussion

K-means is a widely used clustering algorithm due to its simplicity and efficiency. In this activity, K-means was applied to cluster mall customers based on their annual income and spending score. The algorithm successfully segmented customers into distinct groups. The results showed that K-means clustering is sensitive to the initial choice of centroids and the value of K. Different runs of the algorithm can produce varying results due to the random initialization of centroids. The elbow method helped in selecting an optimal K, leading to more consistent and meaningful clusters.