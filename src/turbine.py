#step 1 Load a Data
import pandas as pd
import os
import matplotlib.pyplot as plt


def load_data(file_paths):
    dfs = []
    for file in file_paths:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading file {file}: {e}")
    return pd.concat(dfs, ignore_index=True)

# File paths
file_paths = [
    r"C:\Users\44754\Downloads\Turbine project\inputdata\data_group_1.csv",
    r"C:\Users\44754\Downloads\Turbine project\inputdata\data_group_2.csv",
    r"C:\Users\44754\Downloads\Turbine project\inputdata\data_group_3.csv"
]

# Load data
raw_data = load_data(file_paths)
print(raw_data.head())  # Inspect the first few rows
print(raw_data.info())  # Check for column types and missing values


# step2  Convert 'timestamp' column  from object to datetime
raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'], errors='coerce')

# Drop rows with invalid timestamps
raw_data = raw_data.dropna(subset=['timestamp'])

# Verify the data type of 'timestamp'
print(raw_data.dtypes)
print(raw_data['timestamp'].head())
print(f"Data shape after timestamp parsing: {raw_data.shape}")

print(raw_data['timestamp'].isnull().sum())
print(raw_data['timestamp'].head())  # Inspect parsed timestamps

#step 3 clean the data 
def clean_data(df):
    # Handle missing values (impute with column mean)
    missing_before = df.isnull().sum().sum()
    df.fillna(df.mean(), inplace=True)
    
    # Remove outliers using the interquartile range (IQR)
    Q1 = df['power_output'].quantile(0.25)
    Q3 = df['power_output'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['power_output'] >= lower_bound) & (df['power_output'] <= upper_bound)]
    
    # Return cleaned data and missing data count
    missing_after = missing_before - df.isnull().sum().sum()
    return df, missing_before, missing_after

cleaned_data, missing_before, missing_after = clean_data(raw_data)
print(f"Missing values before cleaning: {missing_before}")
print(f"Missing values handled: {missing_after}")
print(f"Cleaned Data Shape: {cleaned_data.shape}")

 
# step 4 statistics

def calculate_statistics(df):
    summary = df.groupby('turbine_id').agg(
        min_output=('power_output', 'min'),
        max_output=('power_output', 'max'),
        avg_output=('power_output', 'mean'),
        std_output=('power_output', 'std')
    ).reset_index()
    return summary

statistics = calculate_statistics(cleaned_data)
print(f"Summary Statistics:\n{statistics}")

# step 5 identify anomalies

def identify_anomalies(df, summary):
    anomalies = []
    for turbine_id, group in df.groupby('turbine_id'):
        mean = summary.loc[summary['turbine_id'] == turbine_id, 'avg_output'].values[0]
        std = summary.loc[summary['turbine_id'] == turbine_id, 'std_output'].values[0]
        anomaly_mask = (group['power_output'] < mean - 1.5 * std) | (group['power_output'] > mean + 1.5 * std)
        anomalies.append(group[anomaly_mask])
    return pd.concat(anomalies, ignore_index=True)

anomalies = identify_anomalies(cleaned_data, statistics)
print(f"Number of anomalies detected: {anomalies.shape[0]}")




output_dir = r"C:\Users\44754\Downloads\Turbine project\output"
os.makedirs(output_dir, exist_ok=True)

cleaned_data.to_csv(os.path.join(output_dir, "cleaned_data.csv"), index=False)
statistics.to_csv(os.path.join(output_dir, "summary_statistics.csv"), index=False)
anomalies.to_csv(os.path.join(output_dir, "anomalies.csv"), index=False)

print(f"Data saved to: {output_dir}")


# step 6 visualization : 1. Plot anomalies for a specific turbine
def plot_anomalies(turbine_id):
    turbine_data = cleaned_data[cleaned_data['turbine_id'] == turbine_id]
    turbine_anomalies = anomalies[anomalies['turbine_id'] == turbine_id]

    plt.figure(figsize=(12, 6))
    plt.plot(turbine_data['timestamp'], turbine_data['power_output'], label='Power Output', color='blue')
    plt.scatter(turbine_anomalies['timestamp'], turbine_anomalies['power_output'], label='Anomalies', color='red')
    plt.title(f"Power Output with Anomalies (Turbine ID: {turbine_id})")
    plt.xlabel("Timestamp")
    plt.ylabel("Power Output (MW)")
    plt.legend()
    plt.show()

# Call the plot function for a specific turbine
plot_anomalies(1)


# Count anomalies per turbine
anomaly_counts = anomalies.groupby('turbine_id').size().reset_index(name='anomaly_count')
print(anomaly_counts)  # View the anomaly counts for each turbine



# 2.  Bar chart for anomalies per turbine
def plot_anomalies_per_turbine(anomaly_counts):
    plt.figure(figsize=(10, 6))
    plt.bar(anomaly_counts['turbine_id'], anomaly_counts['anomaly_count'], color='red')
    plt.title("Number of Anomalies per Turbine")
    plt.xlabel("Turbine ID")
    plt.ylabel("Number of Anomalies")
    plt.xticks(anomaly_counts['turbine_id'])  # Ensure turbine IDs are visible on the x-axis
    plt.show()

# Call the function
plot_anomalies_per_turbine(anomaly_counts)













