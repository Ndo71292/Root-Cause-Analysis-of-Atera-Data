# PROJECT INSIGHT/INTERPRETATION

## Step 1: Data Collection and Storage

- I created API credentials in the Atera admin center to pull Agents and Alerts data and used two functions (`fetch_agents_from_api` and `fetch_alerts_from_api`) to obtain the data.
- The data is fetched periodically, with a 30-minute interval between each fetching event.

- The functions take two arguments (API key and last timestamp). The last timestamp is useful for keeping track of the time period the data was fetched.

- The fetched data is saved in Amazon S3 buckets (alerts and agents) specified in the main function. See the S3 buckets below.

- The fetched data has columns as follows, but I was interested in a few numeric ones and other new features I created by combining existing features:
  - 'alertid', 'source', 'title', 'severity', 'created4', 'deviceguid5',
  - 'archived', 'alertcategoryid', 'alertmessage', 'devicename',
  - 'customerid10', 'customername11', 'agentid', 'deviceguid13',
  - 'customerid14', 'customername15', 'agentname', 'systemname',
  - 'machinename', 'domainname', 'currentloggedusers', 'modified',
  - 'onlinestatus', 'reportfromip', 'motherboardofmachine',
  - 'memoryofmachine', 'displaygraphics', 'processorcoresnumber', 'vendor',
  - 'title_features', 'alertmessage_features', 'created_year',
  - 'created_month', 'created_day', 'created_hour', 'created_minute'

  ![S3 Buckets](https://github.com/Ndo71292/Images/blob/main/s3%20buckets.png?raw=true)

## Step 2: Data Cleaning

- I created two functions called `process_agents` and `process_alerts`. These functions take three arguments each.
  - For `process_agents(api_key, bucket_name, agents_s3_key)` and for `process_alerts(api_key, bucket_name, alerts_s3_key)`.
  - These functions load the data saved in the S3 buckets, combine it with the newly fetched data from Atera, clean it (drop unnecessary rows, deal with missing values, etc.), and then save the combined data back to S3.

## Step 3: Feature Extraction and New Feature Creation

- As per the project deliverables, the two datasets are supposed to be merged. I used AWS Athena to merge the datasets using the `customerid` as the common key.
  - Athena has permissions to directly access my S3 buckets, making it easy to load the data.
  - I also ran a few queries to check data consistency and integrity.
  - The merged data was saved back to S3 as `MergedData`.

![Athena Join](https://github.com/Ndo71292/Images/blob/main/athena%20join%20.png?raw=true)

## Step 4: Some More Data Cleaning and Validation

- To prepare the merged data for clustering, I went back to Step 2 to perform data cleaning again.
- This time, I used Data Wrangler, which provides a GUI and an option to generate reports on data quality.
- In this step, I created a flow that achieved the following:
  1. One-hot encoding on the categorical features in the merged dataset.
  2. Imputation of missing values in the merged dataset.
  3. Dropped features that had above 50% missing values.
  4. Feature correlation test to pick the best features for clustering analysis.
- The cleaned data was saved back to S3.

![Saving Data to S3](https://github.com/Ndo71292/Images/blob/main/saving%20data%20to%20s3.png?raw=true)

 Below is one of many Data Quality reports generated in Data Wrangler:

![Data Wrangler Insights Report](https://github.com/Ndo71292/Images/blob/main/data-wrangler-insights-report%20(1).png?raw=true)

## Step 4: Model Selection - Why Choose K Means

Choosing K-Means clustering for the Root Cause Analysis (RCA) project has several benefits that align well with the project's objectives and requirements:

1. **Numerical Feature Handling:**
   - K-Means is well-suited for datasets with numerical features, making it appropriate for the machine agents and alerts data in the project. By extracting relevant numerical features from the dataset, K-Means can efficiently partition the data into clusters.

2. **Simplicity and Interpretability:**
   - K-Means is a straightforward and easy-to-understand algorithm, which is beneficial for communication and interpretation of results. The simplicity of cluster shapes (spherical) enhances interpretability, making it easier for stakeholders to grasp the identified patterns and issues.

3. **Scalability:**
   - K-Means is computationally efficient and scalable, making it suitable for the potentially large dataset that may arise from merging machine agents and alerts data. This ensures that the clustering analysis can be performed within reasonable time and computational resources.

4. **Cluster Centroid Representation:**
   - K-Means represents clusters using centroids, making it useful for identifying the central points around which similar alerts and machine agents gather. This can aid in understanding the characteristics of each cluster and facilitate root cause analysis.

5. **Ability to Specify Number of Clusters (K):**
   - The ability to pre-specify the number of clusters (K) in K-Means is advantageous for this project. By choosing an appropriate value of K, the algorithm can group similar alerts and machine agents, helping in the identification of common technical issues more effectively.

6. **Iterative Refinement:**
   - K-Means is amenable to iterative refinement. If the initial results require adjustment or if additional insights are needed, the algorithm can be re-run with different K values or feature sets to enhance the clustering results. This aligns well with the iterative nature of root cause analysis.

7. **Performance Metrics:**
   - K-Means clustering aligns with common performance metrics such as inertia or within-cluster sum of squares, which can be used to assess the quality of the clustering. These metrics can aid in evaluating how well the algorithm is grouping similar alerts and machine agents.

8. **Practical Implementation:**
   - K-Means has been widely used in various applications and is available in popular machine learning libraries. Its practical implementation and availability in tools like scikit-learn make it a convenient choice for this project, facilitating seamless integration into the analysis pipeline.
   - By choosing K-Means clustering, the project aims to leverage its simplicity, numerical feature handling capabilities, scalability, and iterative nature to gain actionable insights into common technical issues and patterns in the machine agents and alerts data, ultimately contributing to the improvement of technical operations and system reliability.

   ## Model Tuning Metrics and Results

For my model tuning, I used two metrics:

1. **Silhouette Score:**
   - **What it measures:** The Silhouette Score measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The score ranges from -1 to 1, where a high value indicates well-defined clusters.
   - **Interpretation:** A higher Silhouette Score indicates better-defined clusters. The best score is 1, and a score near 0 indicates overlapping clusters.
   - **Improvement:** To improve the Silhouette Score, you can try adjusting the number of clusters (`num_clusters`). Experiment with different values to find the optimal number that maximizes the score.

2. **Davies-Bouldin Index:**
   - **What it measures:** The Davies-Bouldin Index quantifies the average similarity between each cluster and its most similar cluster. A lower index suggests better clustering.
   - **Interpretation:** Lower Davies-Bouldin Index values indicate better-defined clusters.
   - **Improvement:** Similar to the Silhouette Score, experimenting with different values of `num_clusters` can help find the configuration that minimizes the Davies-Bouldin Index. Additionally, optimizing feature selection and preprocessing may also improve the index.

I ran my model multiple times, reducing and combining features. I ended up using 3 as the number of clusters. Below is where my scores started and where they ended:

- **At the Start of Model Building:**
  - Silhouette Score: 0.2
  - Davies-Bouldin Index: 1.5

![Model Tuning Start](https://github.com/Ndo71292/Images/blob/main/results_model_tuning.png?raw=true)

- **Final Result:**
  - Silhouette Score: 0.8
  - Davies-Bouldin Index: around 0.7

![Model Tuning Final Result](https://github.com/Ndo71292/Images/blob/main/final_results_model.png?raw=true)

## Step 7: Visualization Of Results

Visualization of the Clusters is the most important stage of my project. It reflects and communicates the objective and deliverables to Stakeholders and users. I put an effort to reduce the number of features as much as possible because I noticed that, initially, my plot was too crowded, making it difficult to separate the points and have a clear picture of what I am trying to achieve.

From the beginning, I had over 100 features that were potentially helpful to my model, but I reduced them to 23 to create room for a good visual result.

Below is what my plot looked like when I ran a feature importance plot to determine the most important features for my model:

There were just too many features to properly interpret the information from the plot. To address this, I combined correlated features into one (e.g., OS-related features, processor-related features, etc).

![Feature Importance](https://github.com/Ndo71292/Images/blob/main/feature%20importance.png?raw=true)

## Step 8: Interpretation of the Visualization

The primary objective of my project was to identify the underlying factors or root causes of technical issues reported through the collected alert and agent data. In production, this model utilizes clustering to pinpoint the specific technical source within the alert/agent data. This functionality significantly streamlines the process for helpdesk technicians, enabling them to visually identify the root cause of a fault. Consequently, this reduces the time invested in investigating and troubleshooting the problem.

Visual representation below illustrates the explanation provided:

- 23 features displayed in the plot/clusters include hardware, software, accessories, timeframes, and users' status and behavior.
- Each point on the cluster represents alert and agent data collected at specific periods by the API.
- Upon receiving the data, it undergoes processing through the model.
- The algorithm maps the data to the correct cluster, indicating which component serves as the root cause of the reported issue.
- For example, the root cause might be identified as vendor-related (vendor on the plot), motherboard-related (motherboardofmachine), user-related (represented by onlinestatus), etc.

![Final Result](https://github.com/Ndo71292/Images/blob/main/final%20result.png?raw=true)

## Project Improvements and Highlights

### Visualization

The visualization generated by K-Means clustering can be hugely improved to tell a compelling story. More details can be added according to the visualization principles to better engage the audience and technicians and even encourage stakeholder buy-in. I would love to create a live dashboard rather than a plot that shows results in real-time. This is possible because my script is programmed to run every 30 minutes and process the results in seconds.

### Model Building

I feel like adding more metrics to measure model performance can be introduced. I also think the model can be improved by adding other models before K-means Clustering. I have added a few points on which models and why:

1. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):**
   - **Why:** DBSCAN is useful for identifying clusters of varying shapes and sizes. It doesn't assume that clusters have a spherical shape, which can be a limitation of K-Means. Including DBSCAN in conjunction with K-Means can help capture more complex structures in the data.

2. **Hierarchical Clustering:**
   - **Why:** Hierarchical clustering can provide a hierarchical decomposition of the data, which may reveal insights about the data's structure. The results of hierarchical clustering can be used to guide the choice of the number of clusters in K-Means or to identify subclusters within larger clusters.

3. **Density-Based Preprocessing:**
   - **Why:** Utilizing techniques to identify and remove outliers or noise before applying K-Means can enhance the quality of clustering. Algorithms like Isolation Forest or Local Outlier Factor can be employed for anomaly detection and noise reduction.
