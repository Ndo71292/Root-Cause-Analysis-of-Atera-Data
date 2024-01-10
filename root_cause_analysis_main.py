import logging
import traceback 
import time
import shutil
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd
import csv
import requests
from datetime import datetime, timezone, timedelta
import json
import string
from dateutil.parser import parse
import re
import plotly.graph_objects as go 
import os
import requests
import traceback
from sklearn.preprocessing import OneHotEncoder
import nltk
import plotly.express as px
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import boto3
from botocore.exceptions import NoCredentialsError
import io
from sklearn.metrics import silhouette_score, davies_bouldin_score


from Functions import (
    process_alerts,process_agents,convert_agent_created_to_datetime,
    fetch_alerts_from_api, investigate_agents_data,print_alerts_csv,identify_relevant_text,
    convert_created_to_datetime,append_agents_to_csv,append_alerts_to_s3,load_agents_data_from_s3,load_alerts_data_from_s3,clustering
)




# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


WAIT_TIME_SECONDS = 1800

while True:
    try:
        process_agents(api_key, bucket_name, agents_s3_key)
        process_alerts(api_key, bucket_name, alerts_s3_key)
        merged_dataset = load_merged_data_from_s3(bucket_name, merged_data_key)

        # Choose the number of clusters
        num_clusters = 3

        # Apply K-Means clustering on combined features
        kmeans_combined = KMeans(n_clusters=num_clusters, random_state=42)
        merged_dataset['combined_cluster'] = kmeans_combined.fit_predict(merged_dataset['combined_matrix'])

        # Evaluate clustering performance
        silhouette_metric = silhouette_score(merged_dataset['combined_matrix'], merged_dataset['combined_cluster'])
        davies_bouldin_metric = davies_bouldin_score(merged_dataset['combined_matrix'], merged_dataset['combined_cluster'])

        # Display metrics
        print(f"Silhouette Score: {silhouette_metric}")
        print(f"Davies-Bouldin Index: {davies_bouldin_metric}")

        # Section 1: Display the pairplot of numeric features with clusters
        numeric_features = merged_dataset.select_dtypes(include=['float64', 'int64']).columns
        features_for_plot = numeric_features.tolist() + ['combined_cluster']
        sns.pairplot(merged_dataset[features_for_plot], hue='combined_cluster', palette='viridis', diag_kind='hist')
        plt.suptitle('Pairplot of Numeric Features with Clusters')
        plt.show()

    except Exception as e:
        print("An error occurred:", e)

    # Calculate the timestamp for the next fetch
    next_fetch_time = datetime.now() + timedelta(minutes=30)

    # Print a message indicating when the next fetch is scheduled
    logger.info(f"Next fetch scheduled at: {next_fetch_time}")

    # Wait for 30 minutes before fetching data again
    time.sleep(WAIT_TIME_SECONDS)



if __name__ == "__main__":
    main()
   

    
