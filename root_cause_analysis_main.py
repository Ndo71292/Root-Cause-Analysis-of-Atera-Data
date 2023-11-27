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



from Functions import (
    process_alerts,process_tickets,process_agents,convert_agent_created_to_datetime,clean_tickets_data,
    fetch_alerts_from_api, investigate_agents_data,print_alerts_csv,preprocess_text_with_tfidf
    ,convert_ticket_created_to_datetime,process_ticket_data,identify_relevant_text,fetch_new_comments_for_ticket,
    convert_created_to_datetime,preprocess_text,update_preprocessed_comments,fetch_tickets_from_api,
    append_agents_to_csv,convert_ticket_created_to_datetime,merge_dataframes_and_join,
    generate_and_show_comparison_visualization,perform_clustering_and_visualize_plotly,append_alerts_to_s3,
    load_tickets_data_from_s3,load_agents_data_from_s3,load_alerts_data_from_s3
)




# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


WAIT_TIME_SECONDS = 1800

merged_dataset = None

def main():
    global merged_dataset 
    
   
    os.environ['AWS_ACCESS_KEY_ID'] = 'xxxxxxxxxxxxxxxxxxxxxxxxx'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'xxxxxxxxxxxxxxxxxxxxxxxxxxx+'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    
    
    
    api_key = 'xxxxxxxxxxxxxxxxxxxxx'
    bucket_name= 'ateradata'
    ticket_s3_key='tickets/tickets.csv'
    #csv_file_tickets = "tickets.csv"  # Specify the correct file path here
    alerts_s3_key = "alerts.csv"  # Specify the correct file path here
    agents_s3_key = "agents.csv"
   
    

    while True:
        try:
            
            
            process_tickets(api_key,bucket_name, ticket_s3_key)
            process_agents(api_key,bucket_name, agents_s3_key)
            process_alerts(api_key,bucket_name,alerts_s3_key)

            

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
   

    
