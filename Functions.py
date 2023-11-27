import logging
import traceback 
import time
import shutil
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA 
import csv
import requests
from datetime import datetime, timezone, timedelta
import json
import string
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from dateutil.parser import parse
import re
import io
import os
import boto3
import botocore
import requests
from sklearn.preprocessing import OneHotEncoder
import nltk
import plotly.express as px
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import boto3
from botocore.exceptions import NoCredentialsError





#Functions for agent data

def fetch_agents_from_api(api_key, last_timestamp=None):
    url = 'https://app.atera.com/api/v3/agents'
    headers = {'Accept': 'application/json', 'X-API-KEY': api_key}

    if last_timestamp:
        # Parse the last_timestamp string into a datetime object
        last_timestamp = parse(last_timestamp)

        # Convert the last_timestamp to ISO 8601 format with timezone info
        last_timestamp_iso = last_timestamp.astimezone(timezone.utc).isoformat()
        params = {"startDate": last_timestamp_iso}
    else:
        params = {}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors (non-2xx responses)
        agents_data = response.json().get("items", [])  # Assuming the API response format is similar
        agents_df = pd.DataFrame(agents_data)
        return agents_df
    except requests.exceptions.RequestException as e:
        print("Failed to fetch agents data from the API:", e)
        return pd.DataFrame()
    
def process_agents(api_key, bucket_name, agents_s3_key):
    print("Reading agents data from S3:", agents_s3_key)

    try:
        # Create an S3 client
        s3_client = boto3.client('s3', region_name='us-east-1')

        # Create an S3 resource
        s3_resource = boto3.resource('s3', region_name=u's-east-1')

        # Load existing agents data from S3
        existing_agents_data = load_agents_data_from_s3(bucket_name, agents_s3_key)

        # Fetch new agent data
        new_agents = fetch_agents_from_api(api_key)

        # Combine existing and new agent data
        combined_agents_data = pd.concat([existing_agents_data, new_agents], ignore_index=True)

        # Convert 'CustomerID' column to int64 data type
        combined_agents_data['CustomerID'] = combined_agents_data['CustomerID'].astype('int64')

        if not combined_agents_data.empty:
            combined_agents_data = convert_agent_created_to_datetime(combined_agents_data)

            # Handle missing values
            missing_values = combined_agents_data.isnull().sum()
            total_rows = len(combined_agents_data)
            missing_percentage = (missing_values / total_rows) * 100
            missing_info = pd.DataFrame({'Column Name': missing_values.index, 'Missing Values': missing_values.values, 'Missing Percentage': missing_percentage.values})

            print("Missing values in combined_agents_data:")
            print(missing_info)

            # Convert 'Created' and 'Modified' columns to datetime
            combined_agents_data['Created'] = pd.to_datetime(combined_agents_data['Created'])
            combined_agents_data['Modified'] = pd.to_datetime(combined_agents_data['Modified'])

            # Extract relevant information from 'CurrentLoggedUsers' column
            combined_agents_data['LoggedUserSince'] = combined_agents_data['CurrentLoggedUsers'].str.extract(r'Since: (.+?)\)')

            # Drop unnecessary columns
            combined_agents_data.drop(columns=[
                'OfficeSP', 'LoggedUserSince', 'MachineID', 'FolderID', 'ComputerDescription',
                'LastPatchManagementReceived', 'MonitoredAgentID', 'Sound', 'ProductName', 'Office',
                'OfficeSP', 'OfficeOEM', 'OfficeSerialNumber', 'OfficeFullVersion', 'ProcessorClock'
            ], inplace=True)

            # Handle missing values after dropping columns
            missing_values_after_drop = combined_agents_data.isnull().sum()
            total_rows_after_drop = len(combined_agents_data)
            missing_percentage_after_drop = (missing_values_after_drop / total_rows_after_drop) * 100
            missing_info_after_drop = pd.DataFrame({'Column Name': missing_values_after_drop.index, 'Missing Values': missing_values_after_drop.values, 'Missing Percentage': missing_percentage_after_drop.values})

            print("Missing values in combined_agents_data after dropping columns:")
            print(missing_info_after_drop)

            # Save the combined and cleaned data to S3
            csv_buffer = io.StringIO()
            combined_agents_data.to_csv(csv_buffer, index=False, encoding='utf-8')
            s3_client.upload_fileobj(io.BytesIO(csv_buffer.getvalue().encode()), bucket_name, agents_s3_key)

            print("Agent data (existing and new) added, cleaned, and missing values handled")

    except Exception as e:
        traceback.print_exc()  # Print the full traceback for the exception
        return combined_agents_data



    
  
    
def convert_agent_created_to_datetime(agents_data):
    for index, agent in agents_data.iterrows():
        created_str = agent["Created"]
        try:
            if isinstance(created_str, str):
                # Parse the timestamp using dateutil.parser
                created_dt = parse(created_str)
                # Convert to UTC if not already in UTC
                if created_dt.tzinfo is None or created_dt.tzinfo.utcoffset(created_dt) is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                agents_data.at[index, "Created"] = created_dt
            else:
                # Handle non-datetime values (e.g., skip or set to None)
                agents_data.at[index, "Created"] = None
        except ValueError:
            print("Error parsing timestamp:", created_str)
    
    # Sort agents_data based on the 'Created' timestamp
    agents_data.sort_values(by="Created", inplace=True)
    
    return agents_data


def append_agents_to_csv(agents_data, bucket_name, agents_s3_key):
    if agents_data is None or agents_data.empty:
        print("No agents data to append.")
        return

    try:
        # Load existing agent data from S3
        existing_agents_data = load_agents_data_from_s3(bucket_name, agents_s3_key)

        # Concatenate the existing data with the new data
        combined_agents_data = pd.concat([existing_agents_data, agents_data], ignore_index=True)

        # Save the combined data back to S3
        upload_to_s3(combined_agents_data, bucket_name, agents_s3_key)

        print("Agents data successfully appended to S3:", agents_s3_key)
    except Exception as e:
        print("Error appending agents to S3:", e)


def load_agents_data_from_s3(bucket_name, agents_s3_key):
    try:
        # Initialize an S3 client
        s3 = boto3.client('s3')
        
        # Load data from S3
        obj = s3.get_object(Bucket=bucket_name, Key=agents_s3_key)
        agents_data = pd.read_csv(obj['Body'], sep=',')
        
        return agents_data
    except Exception as e:
        print(f"Error loading agents data from S3: {e}")
        return pd.DataFrame()

def investigate_agents_data(agents_data):
    try:
        # Step 2: Display the first few rows to check headers
        print("First few rows of the agents data:")
        print(agents_data.head())

        # Step 3: Check for extra rows or data
        num_rows, num_columns = agents_data.shape
        print(f"Number of rows in the agents data: {num_rows}")
        print(f"Number of columns in the agents data: {num_columns}")

        # Step 4: Check data types of columns
        print("\nData types of columns:")
        print(agents_data.dtypes)

        # Step 5: Clean data (if needed)
        # For example, filling missing values with a specific value (e.g., 0)
        # agents_data.fillna(0, inplace=True)

        return agents_data

    except Exception as e:
        print("Error:", e)
        return None

#Functions for alerts
def fetch_alerts_from_api(api_key, last_timestamp=None):
    url = 'https://app.atera.com/api/v3/alerts'  
    headers = {'Accept': 'application/json',
               'X-API-KEY': api_key}

    if last_timestamp:
        # Convert the last_timestamp to ISO 8601 format with timezone info
        last_timestamp_iso = last_timestamp.astimezone(timezone.utc).isoformat()
        params = {"startDate": last_timestamp_iso}
    else:
        params = {}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx and 5xx)
        alerts_data = response.json()
        # Extract the list of alerts from the response
        alerts_list = alerts_data.get("items", [])  
        # Convert the list to a DataFrame
        alerts_df = pd.DataFrame(alerts_list)
        return alerts_df  # Return a DataFrame
    except requests.exceptions.RequestException as e:
        print("Failed to fetch alerts from Atera API:", e)
        return pd.DataFrame()  # Return an empty DataFrame

    
# Function to process alerts from API and save them to S3
def process_alerts(api_key, bucket_name, alerts_s3_key):
    print("Reading alerts data from:", alerts_s3_key)

    try:
        
        # Create an S3 client
        s3_client = boto3.client('s3', region_name='us-east-1')

        # Create an S3 resource
        s3_resource = boto3.resource('s3', region_name=u's-east-1')

        # Load existing alerts data from S3
        existing_alerts_data = load_alerts_data_from_s3(bucket_name, alerts_s3_key)

        # Fetch new alert data
        new_alerts_data = fetch_alerts_from_api(api_key)

        # Combine existing and new alerts data
        combined_alerts_data = pd.concat([existing_alerts_data, new_alerts_data], ignore_index=True)

        # Convert 'CustomerID' column to int64 data type
        combined_alerts_data['CustomerID'] = pd.to_numeric(combined_alerts_data['CustomerID'], errors='coerce', downcast='integer')

        # Filter out rows with non-integer 'CustomerID'
        combined_alerts_data = combined_alerts_data.dropna(subset=['CustomerID']).astype({'CustomerID': 'int64'})

        # Convert 'Created' to datetime with UTC timezone
        combined_alerts_data['Created'] = pd.to_datetime(combined_alerts_data['Created'], format='%Y-%m-%dT%H:%M:%S', errors='coerce').dt.tz_convert('UTC')

        # Check and handle missing values in 'CustomerID' and 'CustomerName'
        customer_name_to_id = dict(zip(combined_alerts_data['CustomerName'], combined_alerts_data['CustomerID']))
        customer_id_to_name = dict(zip(combined_alerts_data['CustomerID'], combined_alerts_data['CustomerName']))
        combined_alerts_data['CustomerID'].fillna(combined_alerts_data['CustomerName'].map(customer_name_to_id), inplace=True)
        combined_alerts_data['CustomerName'].fillna(combined_alerts_data['CustomerID'].map(customer_id_to_name), inplace=True)

        # Handle missing values and set a placeholder for 'AlertCategoryID'
        placeholder_value = 'Unknown'
        combined_alerts_data['AlertCategoryID'].fillna(placeholder_value, inplace=True)

        # Check for negative values in numeric columns
        numeric_columns = ['AlertID', 'TicketID', 'PollingCyclesCount']
        numeric_columns = [col for col in numeric_columns if combined_alerts_data[col].dtype == 'int64']  # Adjust the dtype if necessary
        negative_values = combined_alerts_data[numeric_columns].lt(0).any()

        # Drop rows with missing values in specific columns
        columns_to_drop = ['Code', 'SnoozedEndDate', 'AdditionalInfo', 'ArchivedDate', 'FolderID', 'PollingCyclesCount', 'TicketID']
        combined_alerts_data.drop(columns=columns_to_drop, inplace=True)

        # Save the combined and processed data back to S3
        csv_buffer = io.StringIO()
        combined_alerts_data.to_csv(csv_buffer, index=False, encoding='utf-8')

        # Upload the data to S3
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.upload_fileobj(io.BytesIO(csv_buffer.getvalue().encode()), bucket_name, alerts_s3_key)

        print(combined_alerts_data.head())
        print('Alerts data saved to S3')

    except Exception as e:
        traceback.print_exc()
        return combined_alerts_data


def append_alerts_to_s3(alerts_data, bucket_name, alerts_s3_key):
    if not alerts_data.empty:
        fieldnames = ["AlertID", "Source", "Title", "Severity", "Created", "DeviceGuid",
                      "Archived", "AlertCategoryID", "TicketID", "AlertMessage",
                      "DeviceName", "CustomerID", "CustomerName"]
        
        try:
            # Prepare the data as a CSV string
            csv_buffer = io.StringIO()
            alerts_data.to_csv(csv_buffer, index=False, quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8')

            # Upload the data to S3
            s3_client = boto3.client('s3', region_name='us-east-1')
            s3_client.put_object(Bucket=bucket_name, Key=alerts_s3_key, Body=csv_buffer.getvalue())

            print("Alerts data successfully appended to S3.")
        except Exception as e:
            print("Error appending alerts to S3:", e)
    else:
        print("No data to append.")
        
def print_alerts_csv(csv_file):
    with open(csv_file, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            print(row)
            

def convert_created_to_datetime(alerts_data):
    for index, alert in alerts_data.iterrows():
        created_str = alert["Created"]
        try:
            # Parse the timestamp using dateutil.parser
            created_dt = parse(created_str)
            # Convert to UTC if not already in UTC
            if created_dt.tzinfo is None or created_dt.tzinfo.utcoffset(created_dt) is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            alerts_data.at[index, "Created"] = created_dt
        except ValueError:
            print("Error parsing timestamp:", created_str)
            
    # Sort alerts_data based on the 'Created' timestamp
    alerts_data.sort_values(by="Created", inplace=True)
    
    return alerts_data

# Function to load alerts data from S3
def load_alerts_data_from_s3(bucket_name, alerts_s3_key):
    try:
        # Create an S3 client
        s3_client = boto3.client('s3', region_name='us-east-1')

        # Load data from S3
        obj = s3_client.get_object(Bucket=bucket_name, Key=alerts_s3_key)
        alerts_data = pd.read_csv(obj['Body'])

        return alerts_data
    except NoCredentialsError:
        print("No AWS credentials found. Make sure you have configured your AWS CLI or SDK properly.")
    except Exception as e:
        print(f"Error loading data from S3: {e}")
        return pd.DataFrame()
           
# Function to load merged data from S3 by Customer ID (mergedbycid)
def load_mergedbycid_data_from_s3(bucket_name, merged_bycid_key):
    try:
        # Create an S3 client
        s3_client = boto3.client('s3', region_name='us-east-1')

        # Load data from S3
        obj = s3_client.get_object(Bucket=bucket_name, Key=merged_bycid_key)
        mergedbycid_data = pd.read_csv(obj['Body'])

        return mergedbycid_data  # Return the loaded data
    except Exception as e:
        print(f"Error loading data from S3: {e}")
        return pd.DataFrame()

# Function to load merged data from S3 by Device Guid (mergedbydg)
def load_mergedbydg_data_from_s3(bucket_name, merged_bydg_key):
    try:
        # Create an S3 client
        s3_client = boto3.client('s3', region_name='us-east-1')

        # Load data from S3
        obj = s3_client.get_object(Bucket=bucket_name, Key=merged_bydg_key)
        merged_bydg_data = pd.read_csv(obj['Body'])

        return merged_bydg_data  # Return the loaded data
    except Exception as e:
        print(f"Error loading data from S3: {e}")
        return pd.DataFrame()









                

