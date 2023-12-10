import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
import pandas as pd

def calculate_distance_matrix(df):
    # Create a DataFrame with unique toll locations
    unique_tolls = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    distance_matrix = pd.DataFrame(index=unique_tolls, columns=unique_tolls)
    
    # Initialize the distance matrix with zeros on the diagonal
    distance_matrix.values[[range(len(distance_matrix))]*2] = 0
    
    # Iterate through rows of the DataFrame to calculate cumulative distances
    for index, row in df.iterrows():
        start_toll = row['id_start']
        end_toll = row['id_end']
        distance = row['distance']

        # Update cumulative distances in both directions
        distance_matrix.loc[start_toll, end_toll] += distance
        distance_matrix.loc[end_toll, start_toll] += distance
    
    return distance_matrix

# Example usage:
# Assuming df is your DataFrame loaded from dataset-3.csv
df = pd.read_csv(r'C:\Users\Sandy Salim\Desktop\AssesmentProject\MapUp-Data-Assessment-F\datasets\dataset-3.csv')
distance_matrix_result = calculate_distance_matrix(df)
print(distance_matrix_result)





def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
import pandas as pd

def unroll_distance_matrix(distance_matrix):
    # Create a list to store the unrolled data
    unrolled_data = []

    # Iterate through the rows and columns of the distance matrix
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            # Skip diagonal entries (id_start == id_end)
            if id_start != id_end:
                distance = distance_matrix.loc[id_start, id_end]
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    # Create a DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df

# Example usage:
# Assuming distance_matrix_result is the DataFrame obtained from Question 1
unrolled_distance_df = unroll_distance_matrix(distance_matrix_result)
print(unrolled_distance_df)




def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
import pandas as pd

def find_ids_within_ten_percentage_threshold(df, reference_value):
    # Calculate the average distance for the reference value
    reference_avg_distance = df[df['id_start'] == reference_value]['distance'].mean()

    # Calculate the threshold range (within 10%)
    lower_threshold = reference_avg_distance * 0.9
    upper_threshold = reference_avg_distance * 1.1

    # Filter values within the threshold range and return a sorted list
    filtered_ids = df[(df['id_start'] != reference_value) & 
                      (df['distance'] >= lower_threshold) & 
                      (df['distance'] <= upper_threshold)]['id_start'].unique()

    sorted_filtered_ids = sorted(filtered_ids)

    return sorted_filtered_ids

# Example usage:
# Assuming unrolled_distance_df is the DataFrame obtained from Question 2
reference_value = 1  # Replace with the desired reference value
result_ids = find_ids_within_ten_percentage_threshold(unrolled_distance_df, reference_value)
print(result_ids)



def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
import pandas as pd

def calculate_toll_rate(df):
    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Create new columns for toll rates based on vehicle types
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df

# Example usage:
# Assuming unrolled_distance_df is the DataFrame obtained from Question 2
df_with_toll_rates = calculate_toll_rate(unrolled_distance_df)
print(df_with_toll_rates)





def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
import pandas as pd
from datetime import datetime, time

def calculate_time_based_toll_rates(df):
    # Define time ranges for weekdays and weekends
    weekday_time_ranges = [(time(0, 0, 0), time(10, 0, 0)),
                           (time(10, 0, 0), time(18, 0, 0)),
                           (time(18, 0, 0), time(23, 59, 59))]

    weekend_time_range = (time(0, 0, 0), time(23, 59, 59))

    # Create new columns for start_day, start_time, end_day, and end_time
    df['start_day'] = df['start_datetime'].dt.day_name()
    df['end_day'] = df['end_datetime'].dt.day_name()
    df['start_time'] = df['start_datetime'].dt.time
    df['end_time'] = df['end_datetime'].dt.time

    # Apply discount factors based on time ranges for weekdays and weekends
    for start_time, end_time in weekday_time_ranges:
        weekday_mask = (df['start_datetime'].dt.time >= start_time) & (df['end_datetime'].dt.time <= end_time) & (df['start_datetime'].dt.weekday < 5)
        df.loc[weekday_mask, ['moto', 'car', 'rv', 'bus', 'truck']] *= 0.8

    weekend_mask = (df['start_datetime'].dt.weekday >= 5)
    df.loc[weekend_mask, ['moto', 'car', 'rv', 'bus', 'truck']] *= 0.7

    return df

# Example usage:
# Assuming df_with_time_intervals is the DataFrame obtained from Question 3
df_with_time_based_toll_rates = calculate_time_based_toll_rates(df_with_time_intervals)
print(df_with_time_based_toll_rates)



