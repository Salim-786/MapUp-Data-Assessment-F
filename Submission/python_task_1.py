import pandas as pd


def generate_car_matrix(df):
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Pivot the DataFrame to create the desired matrix
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car')
    
    # Fill NaN values with 0 and set diagonal values to 0
    car_matrix = car_matrix.fillna(0)
    car_matrix.values[[range(len(car_matrix))]*2] = 0


    return car_matrix

df = pd.read_csv(r'C:\Users\Sandy Salim\Desktop\AssesmentProject\MapUp-Data-Assessment-F\datasets\dataset-1.csv')
result_matrix = generate_car_matrix(df)
print(result_matrix)


import pandas as pd
import numpy as np
def get_type_count(df):
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Add a new categorical column 'car_type' based on 'car' values
    conditions = [
        (df['car'] <= 15),
        (df['car'] > 15) & (df['car'] <= 25),
        (df['car'] > 25)
    ]
    choices = ['low', 'medium', 'high']
    df['car_type'] = pd.Series(np.select(conditions, choices, default=''), dtype='category')

    # Calculate the count of occurrences for each 'car_type' category
    type_counts = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    sorted_type_counts = dict(sorted(type_counts.items()))

    return sorted_type_counts

# Assuming df is your DataFrame loaded from dataset-1.csv
# Example usage:
df = pd.read_csv(r'C:\Users\Sandy Salim\Desktop\AssesmentProject\MapUp-Data-Assessment-F\datasets\dataset-1.csv')
result = get_type_count(df)
print(result)



import pandas as pd
def get_bus_indexes(df):
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Identify indices where 'bus' values are greater than twice the mean value
    mean_bus_value = df['bus'].mean()
    bus_indexes = df[df['bus'] > 2 * mean_bus_value].index.tolist()
    
    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes



# Assuming df is your DataFrame loaded from dataset-1.csv
# Example usage for get_bus_indexes:
df = pd.read_csv(r'C:\Users\Sandy Salim\Desktop\AssesmentProject\MapUp-Data-Assessment-F\datasets\dataset-1.csv')
bus_indices = get_bus_indexes(df)
print("Bus Indices:", bus_indices)





import pandas as pd
def filter_routes(df):
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
        # Filter routes based on the average of 'truck' column
        routes_above_threshold = df.groupby('route')['truck'].mean() > 7
        filtered_routes = routes_above_threshold[routes_above_threshold].index.tolist()

        # Sort the list of values in ascending order
        filtered_routes.sort()

        return filtered_routes
    
# Example usage for filter_routes:
df = pd.read_csv(r'C:\Users\Sandy Salim\Desktop\AssesmentProject\MapUp-Data-Assessment-F\datasets\dataset-1.csv')
filtered_route_values = filter_routes(df)
print("Filtered Routes:", filtered_route_values)



def multiply_matrix(result_matrix):
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Copy the DataFrame to avoid modifying the original DataFrame
    modified_matrix = result_matrix.copy()

    # Apply the specified logic to modify values
    modified_matrix[modified_matrix > 20] *= 0.75
    modified_matrix[modified_matrix <= 20] *= 1.25

    # Round values to 1 decimal place
    modified_matrix = modified_matrix.round(1)

    return modified_matrix

# Example usage:
# Assuming result_matrix is the DataFrame obtained from Question 1
modified_result_matrix = multiply_matrix(result_matrix)
print(modified_result_matrix)




def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
import pandas as pd


def verify_timestamp_completeness(df):
    # Combine 'startDay' and 'startTime' columns to create a datetime column
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])

    # Combine 'endDay' and 'endTime' columns to create a datetime column
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    # Calculate the duration for each unique ('id', 'id_2') pair
    df['duration'] = df['end_datetime'] - df['start_datetime']

    # Check if duration covers a full 24-hour period and spans all 7 days of the week
    completeness_check = (
        (df['duration'] >= pd.Timedelta(hours=24)) &
        (df['start_datetime'].dt.time == pd.Timestamp('00:00:00').time()) &
        (df['end_datetime'].dt.time == pd.Timestamp('23:59:59').time()) &
        (df['start_datetime'].dt.dayofweek == 0) &  # Monday
        (df['end_datetime'].dt.dayofweek == 6)     # Sunday
    )

    # Create a boolean series with multi-index ('id', 'id_2')
    completeness_series = completeness_check.groupby(['id', 'id_2']).all()

    return completeness_series

# Example usage:
# Assuming df is your DataFrame loaded from dataset-2.csv
df = pd.read_csv(r'C:\Users\Sandy Salim\Desktop\AssesmentProject\MapUp-Data-Assessment-F\datasets\dataset-2.csv')
completeness_result = verify_timestamp_completeness(df)
print(completeness_result
