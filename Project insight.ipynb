{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "8eecaa1f",
   "metadata": {},
   "source": [
    "# PROJECT INSIGHT/INTERPRETATION"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f3916113",
   "metadata": {},
   "source": [
    "## Step 1: Data Collection and Storage\n",
    "\n",
    "- I created API credentials in Atera admin center for me to be able to pull Agents and Alerts data and used 2 Functions (fetch_agents_from_api and fetch_alerts_from_api) to get the data.\n",
    "\n",
    "- The functions take 2 arguments (api key and last timestamp).The last timestamp is useful to keep track of the time period the data was fetched. \n",
    "\n",
    "- The data fetched is saved in Amazon S3 buckets (alerts and agents) specified in the main function. see the s3 buckets below\n",
    "\n",
    "- The data fetched have columns as follows but i was interested in a few numeric ones and other new features i created from combining existing features .  \n",
    "- 'alertid', 'source', 'title', 'severity', 'created4', 'deviceguid5',\n",
    "       'archived', 'alertcategoryid', 'alertmessage', 'devicename',\n",
    "       'customerid10', 'customername11', 'agentid', 'deviceguid13',\n",
    "       'customerid14', 'customername15', 'agentname', 'systemname',\n",
    "       'machinename', 'domainname', 'currentloggedusers', 'modified',\n",
    "       'onlinestatus', 'reportfromip', 'motherboardofmachine',\n",
    "       'memoryofmachine', 'displaygraphics', 'processorcoresnumber', 'vendor',\n",
    "       'title_features', 'alertmessage_features', 'created_year',\n",
    "       'created_month', 'created_day', 'created_hour', 'created_minute"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e35c26b4",
   "metadata": {},
   "source": [
    "![alt text](https://github.com/Ndo71292/Images/blob/main/s3%20buckets.png?raw=true)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "709db841",
   "metadata": {},
   "source": [
    "## Step 2 Data Cleaning \n",
    "- I created 2 functions called process_agents and process_alerts. These functions take 3 arguments each. \n",
    "\n",
    "- For process_agents (api_key, bucket_name, agents_s3_key ) and for process_alerts(api_key, bucket_name, alerts_s3_key). \n",
    "\n",
    "- These functions loads the data saved in the s3 buckets and combines it with the newly fetched data from atera  and the cleans it (drop unnecesary rows, deals with missing values etc) the save the combined data back to s3."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "28abf33f",
   "metadata": {},
   "source": [
    "## Step 3 Feature Extraction and New Feauture Creation\n",
    "- As per the project deliverables, the two datasets are supposed to be merged. I used AWS Athena to merge the datasets using the customerid as the common key. see below\n",
    "- Athena has permissions to directly access my s3 buckets so it was easy to load the data.\n",
    "- I also ran a few queries to check data consistency and integrity\n",
    "- The merged data was saved back to s3 as Mergeddata"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c2af3664",
   "metadata": {},
   "source": [
    "![alt text](https://github.com/Ndo71292/Images/blob/main/athena%20join%20.png?raw=true)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ba0dcb5f",
   "metadata": {},
   "source": [
    "## Step 4 Some More Data Cleaning and Validation\n",
    "\n",
    "- To prepared the merged data for clustering i went back to step 2 to do the data cleaning again.\n",
    "- This time i used Data Wrangler which gives a GUI shown below and an option to generate reports on data quality.\n",
    "\n",
    "- In this step i a flow that achieved the following\n",
    "1. one hot encoding on the categorical features in the merged data set \n",
    "2. imputation of missing values in the merged data set\n",
    "3. dropped featues that had above 50% missing values\n",
    "4. feature correlation test to pick the best features for clustering analysis\n",
    "\n",
    "- The cleaned data was saved back to s3\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e9a1c9d0",
   "metadata": {},
   "source": [
    "![alt text](https://github.com/Ndo71292/Images/blob/main/saving%20data%20to%20s3.png?raw=true)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "688d7b3b",
   "metadata": {},
   "source": [
    "Below is one of many Data Quality reports generated in data Wrangler"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ba0e2508",
   "metadata": {},
   "source": [
    "![alt text](https://github.com/Ndo71292/Images/blob/main/data-wrangler-insights-report%20(1).png?raw=true)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7300bb68",
   "metadata": {},
   "source": [
    "## Step 4: Model Selection . Why chose K Means \n",
    "    \n",
    "- Choosing K-Means clustering for the Root Cause Analysis (RCA) project has several benefits that align well with the project's objectives and requirements:\n",
    "\n",
    "1. Numerical Feature Handling:\n",
    "\n",
    "- K-Means is well-suited for datasets with numerical features, making it appropriate for the machine agents and alerts data in the project. By extracting relevant numerical features from the dataset, K-Means can efficiently partition the data into clusters.\n",
    "\n",
    "\n",
    "2. Simplicity and Interpretability:\n",
    "\n",
    "- K-Means is a straightforward and easy-to-understand algorithm, which is beneficial for communication and interpretation of results. The simplicity of cluster shapes (spherical) enhances interpretability, making it easier for stakeholders to grasp the identified patterns and issues.\n",
    "\n",
    "\n",
    "3. Scalability:\n",
    "\n",
    "- K-Means is computationally efficient and scalable, making it suitable for the potentially large dataset that may arise from merging machine agents and alerts data. This ensures that the clustering analysis can be performed within reasonable time and computational resources.\n",
    "\n",
    "4. Cluster Centroid Representation:\n",
    "\n",
    "- K-Means represents clusters using centroids, making it useful for identifying the central points around which similar alerts and machine agents gather. This can aid in understanding the characteristics of each cluster and facilitate root cause analysis.\n",
    "\n",
    "5. Ability to Specify Number of Clusters (K):\n",
    "\n",
    "- The ability to pre-specify the number of clusters (K) in K-Means is advantageous for this project. By choosing an appropriate value of K, the algorithm can group similar alerts and machine agents, helping in the identification of common technical issues more effectively.\n",
    "\n",
    "6. Iterative Refinement:\n",
    "\n",
    "- K-Means is amenable to iterative refinement. If the initial results require adjustment or if additional insights are needed, the algorithm can be re-run with different K values or feature sets to enhance the clustering results. This aligns well with the iterative nature of root cause analysis.\n",
    "\n",
    "7. Performance Metrics:\n",
    "\n",
    "- K-Means clustering aligns with common performance metrics such as the inertia or within-cluster sum of squares, which can be used to assess the quality of the clustering. These metrics can aid in evaluating how well the algorithm is grouping similar alerts and machine agents.\n",
    "\n",
    "8. Practical Implementation:\n",
    "\n",
    "- K-Means has been widely used in various applications and is available in popular machine learning libraries. Its practical implementation and availability in tools like scikit-learn make it a convenient choice for this project, facilitating seamless integration into the analysis pipeline.\n",
    "- By choosing K-Means clustering, the project aims to leverage its simplicity, numerical feature handling capabilities, scalability, and iterative nature to gain actionable insights into common technical issues and patterns in the machine agents and alerts data, ultimately contributing to the improvement of technical operations and system reliability."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7f088c4b",
   "metadata": {},
   "source": [
    "## Step 6: Model Building and Tuning\n",
    "\n",
    "For my model Tuning i used 2 matrics as below. I provided i few pointers to explain what each measures and how to improve it.\n",
    "\n",
    "1. Silhouette Score:\n",
    "- What it measures: The Silhouette Score measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The score ranges from -1 to 1, where a high value indicates well-defined clusters.\n",
    "\n",
    "- Interpretation: A higher Silhouette Score indicates better-defined clusters. The best score is 1, and a score near 0 indicates overlapping clusters.\n",
    "\n",
    "- Improvement: To improve the Silhouette Score, you can try adjusting the number of clusters (num_clusters). Experiment with different values to find the optimal number that maximizes the score.\n",
    "\n",
    "\n",
    "2. Davies-Bouldin Index:\n",
    "- What it measures: The Davies-Bouldin Index quantifies the average similarity between each cluster and its most similar cluster. A lower index suggests better clustering.\n",
    "\n",
    "- Interpretation: Lower Davies-Bouldin Index values indicate better-defined clusters.\n",
    "\n",
    "- Improvement: Similar to the Silhouette Score, experimenting with different values of num_clusters can help find the configuration that minimizes the Davies-Bouldin Index. Additionally, optimizing feature selection and preprocessing may also improve the index.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "da115a6b",
   "metadata": {},
   "source": [
    "I ran my model multiple times reducing and combining features. I ended up using 3 as the number of clusters Below is where my score started to where they ended"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "065dc83d",
   "metadata": {},
   "source": [
    "At Start of Model Building my Silhouette Score was 0.2\n",
    "and Davies-Bouldin Index value was 1.5"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2ce45dec",
   "metadata": {},
   "source": [
    "![alt text](https://github.com/Ndo71292/Images/blob/main/results_model_tuning.png?raw=true)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "59b75ce8",
   "metadata": {},
   "source": [
    "Final Result:  Silhouette Score was 0.8\n",
    "and Davies-Bouldin Index value around 0.7"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6b5b6858",
   "metadata": {},
   "source": [
    "![alt text](https://github.com/Ndo71292/Images/blob/main/final_results_model.png?raw=true)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "edc74cb9",
   "metadata": {},
   "source": [
    "## Step 7: Visualization Of Results "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "efb448bc",
   "metadata": {},
   "source": [
    " Visualization of the Clusters is the most important stage of my project. It reflects and communicates the objective and deliverables to Stakeholders and users. I put an effort to reduce the number of features as much as possble because i noticed that, at first my plot was too crowded and it was diffict to seperate the points to have a clear picture of what i am trying to achieve. \n",
    "\n",
    "From the beginning i had over 100 features that was potentially helpful to my model but i reduced them to 23 to create room for good visual result.\n",
    "\n",
    "Below is what my plot was looking like when i run a feature importance plot to determine the most impoertant features for my model\n",
    "\n",
    "There were just too many features to properly interpret the information from the plot.\n",
    "To address this i combined correlated features into one eg os related features, processor related features etc"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b4efa4ec",
   "metadata": {},
   "source": [
    "![alt text](https://github.com/Ndo71292/Images/blob/main/feature%20importance.png?raw=true)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c929d663",
   "metadata": {},
   "source": [
    "## Step 8 Interpretation of the Visualization\n",
    "The primary objective of my project was to identify the underlying factors or root causes of technical issues reported through the collected alert and agent data. In production, this model utilizes clustering to pinpoint the specific technical source within the alert/agent data. This functionality significantly streamlines the process for helpdesk technicians, enabling them to visually identify the root cause of a fault. Consequently, this reduces the time invested in investigating and troubleshooting the problem.\n",
    "\n",
    "Visual representation below illustrates the explanation provided.\n",
    "- 23 features displayed in the plot/clusters include hardware, software, accessories, timeframes, and users' status and behavior.\n",
    "- Each point on the cluster represents alert and agent data collected at specific periods by the API.\n",
    "- Upon receiving the data, it undergoes processing through the model.\n",
    "- The algorithm maps the data to the correct cluster, indicating which component serves as the root cause of the reported issue.\n",
    "- For example , the root cause might be identified as vendor related(vendor on the plot), motherboard related(motherboardofmachine), user related(represented by onlinestatus), etc"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b164af94",
   "metadata": {},
   "source": [
    "![alt text](https://github.com/Ndo71292/Images/blob/main/final%20result.png?raw=true)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ea0019b7",
   "metadata": {},
   "source": [
    "## Project Improvements and Highlights"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ae44c2e3",
   "metadata": {},
   "source": [
    "### Visualization\n",
    "- The visualization generation by k means clustering can be hugely improved to tell a compelling story. More details can be added according to the visualization principles to better engage the audience and technicians and even encourage stakeholder buy in. I Want love to create a live dashbord rather than a plot that shows results in real time. This is possible because my script is programed to run every 30 minutes and process the results in seconds. \n",
    "\n",
    "## Model Building\n",
    "I feel like adding more metrics to measure model performance can be introduced. I also think the model can be improved by adding other models before K means Clustering. I have added a few points of which models and why.\n",
    "\n",
    "1. DBSCAN (Density-Based Spatial Clustering of Applications with Noise):\n",
    "\n",
    "Why: DBSCAN is useful for identifying clusters of varying shapes and sizes. It doesn't assume that clusters have a spherical shape, which can be a limitation of K-Means. Including DBSCAN in conjunction with K-Means can help capture more complex structures in the data.\n",
    "\n",
    "2. Hierarchical Clustering:\n",
    "\n",
    "Why: Hierarchical clustering can provide a hierarchical decomposition of the data, which may reveal insights about the data's structure. The results of hierarchical clustering can be used to guide the choice of the number of clusters in K-Means or to identify subclusters within larger clusters.\n",
    "\n",
    "3. Density-Based Preprocessing:\n",
    "\n",
    "Why: Utilizing techniques to identify and remove outliers or noise before applying K-Means can enhance the quality of clustering. Algorithms like Isolation Forest or Local Outlier Factor can be employed for anomaly detection and noise reduction."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9ec88149",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
