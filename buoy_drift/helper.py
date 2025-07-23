"""
******************************************************************************

 Project:    Polar Cal/Val
 Purpose:    Helper function to buoy_drift
 Author:     Brendon Gory, brendon.gory@noaa.gov
                           brendon.gory@colostate.edu
             Data Science Application Specialist (Research Associate II)
             at CSU CIRA
 Supervisor: Dr. Prasanjit Dash, prasanjit.dash@noaa.gov
                               prasanjit.dash@colostate.edu
             CSU CIRA Research Scientist III
             (Program Innovation Scientist)
******************************************************************************
Copyright notice
         NOAA STAR SOCD and Colorado State Univ CIRA
         2025, Version 1.0.0
         POC: Brendon Gory (brendon.gory@noaa.gov)

 Permission is hereby granted, free of charge, to any person obtaining a
 copy of this software and associated documentation files (the "Software"),
 to deal in the Software without restriction, including without limitation
 the rights to use, copy, modify, merge, publish, distribute, sublicense,
 and/or sell copies of the Software, and to permit persons to whom the
 Software is furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included
 in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 DEALINGS IN THE SOFTWARE.
"""
from typing import Tuple


def fetch_iabp_acronyms(url="https://iabp.apl.uw.edu/Acronyms.html"):
    """
    Fetch and parse buoy-related acronyms from the IABP
    (International Arctic Buoy Program) website.

    This function retrieves the HTML content from the provided URL,
    extracts all acronym tables on the page, and returns the acronyms
    organized by category.

    Args:
        url (str, optional): URL of the IABP Acronyms page.
            Defaults to "https://iabp.apl.uw.edu/Acronyms.html".

    Returns:
        dict: Dictionary where each key is a category name (e.g., 
        "Buoy Owner Acronyms", "Buoy Type Acronyms") and the value is 
        another dictionary mapping acronyms to their full meanings.
        Example:
        {
            "Buoy Owner Acronyms": {
                "NOAA": "National Oceanic and Atmospheric Administration",
                "JMA": "Japan Meteorological Agency",
                ...
            },
            "Buoy Type Acronyms": {
                "SVP": "Surface Velocity Program Drifter",
                ...
            },
            ...
        }

    Raises:
        Exception: If the HTTP request to the URL fails.
        ValueError: If a category name is not found in a table.

    Dependencies:
        - `requests` for downloading HTML content.
        - `bs4` (BeautifulSoup) for parsing the HTML structure.

    Example:
        >>> acronyms = fetch_iabp_acronyms()
        >>> acronyms["Buoy Type Acronyms"]["SVP"]
        'Surface Velocity Program Drifter'
    """
    
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch data from {url}. "
            f"HTTP Status: {response.status_code}"
            )

    soup = BeautifulSoup(response.text, 'html.parser')
    tables = soup.find_all("table")

    acronym_data = {}

    for table in tables:
        # Get the table name
        category = ''
        rows = table.find_all("th", colspan=2)
        if len(rows) == 0:
            raise ValueError(f'No category found in {rows}')
        category = rows[0].get_text(strip=True)
        if category == '':
            raise ValueError(f'No category found in {rows}')
        
        
        # Get acronymns and meanings
        acronyms_dict = {}
        rows = table.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) > 0:
                for idx, col in enumerate(cols):
                    if idx == 0:
                        acronym = col.get_text(strip=True)
                    elif idx == 1:
                        meaning = col.get_text(strip=True)
                acronyms_dict[acronym] =  meaning
            
        acronym_data[category] = acronyms_dict

    return acronym_data

  
def compute_bearing(
    lat1: float, lon1: float,
    lat2: float, lon2: float 
    ) -> Tuple[float, float]: # Returns a tuple: (fwd_azimuth, distance)

    """
    Calculate the daily drift between two points (start and end) based on 
    their latitude and longitude.

    Parameters:
        lat1 (float): Starting latitude(s), can be a single float
                      or a list of floats
        lon1 (float): Starting longitude(s), can be a single float
                      or a list of floats 
        lat2 (float): Ending latitude(s), can be a single float
                      or a list of floats
        lon2 (float): Ending longitude(s), can be a single float
                      or a list of floats  
    Returns: 
        fwd_azimuth (float): Forward azimuth in degrees (0 to 360),
                             measured clockwise from true north
        distance (float): Great circle distance between the two points
                          in meters
    Reference:
        https://pyproj4.github.io/pyproj/stable/api/geod.html#pyproj.Geod.inv
        
    Author:
        Sun Bak-Hospital, sun.bak-hospital@noaa.gov
    """

    from pyproj import Geod 

    # Initialize a geodetic object using the WGS84 ellipsoid
    geod = Geod(ellps='WGS84')

    # Calculate azimuth and distance using geod.inv method
    # Note: Arguments order is (lon1, lat1, lon2, lat2) 
    # as required by pyproj.Geod.inv
    fwd_azimuth, _ , distance = geod.inv(lon1, lat1, lon2, lat2)
    
    # Return the calculated forward azimuth, back azimuth, and distance
    return fwd_azimuth, distance 


def download_erddap_buoy_csv(user_args):
    """
    Download buoy data CSV from NOAA ERDDAP within the specified date range.

    This function constructs a URL based on `start_date` and `end_date`
    from the `user_args` dictionary, and downloads a CSV file from the
    NOAA PolarWatch ERDDAP server (`iabpv2_buoys` dataset). If a local file
    matching the date range already exists, the function skips downloading.

    Args:
        user_args (dict): Dictionary containing script configuration
        parameters.
            Expected keys:
                - 'start_date' (str): Start date in 'YYYYMMDD' format.
                - 'end_date' (str): End date in 'YYYYMMDD' format.
                - 'infile_rootname' (str): Root name for the output CSV file.
                - 'work_path' (str): Directory where the file will be saved.

    Returns:
        None

    Side Effects:
        - Writes a CSV file to disk with name format:
          `<infile_rootname>_<start_date>_<end_date>.csv`
        - Prints status messages and a download progress bar.
        - Raises an error if the file could not be downloaded.

    Raises:
        ValueError: If the HTTP request fails or returns a non-200 status code.
        requests.RequestException: If there is a problem with the request.

    Example:
        >>> user_args = {
                'start_date': '20250101',
                'end_date': '20250115',
                'infile_rootname': 'ERDDAP_IABP',
                'work_path': './data'
            }
        >>> download_erddap_buoy_csv(user_args)
        # Downloads file to ./data/ERDDAP_IABP_2025-01-01_2025-01-15.csv
    """
    
    import requests
    import os
    from datetime import datetime
    from tqdm import tqdm
    
    # By using headers, it sometimes allows the web site to 
    # authenticate the request
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/'
            '537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )
    }
    
    start_date = datetime.strptime(f"{user_args['start_date']}", "%Y%m%d")    
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = datetime.strptime(f"{user_args['end_date']}", "%Y%m%d")
    end_date = end_date.strftime('%Y-%m-%d')
    
    local_file_path = os.path.join(
        user_args['work_path'],
        f"{user_args['infile_rootname']}_{start_date}_{end_date}.csv"
        )
    if os.path.exists(local_file_path):
        print("Found local copy. Not downloading.")
        return

    url = (
        "https://polarwatch.noaa.gov/erddap/tabledap/iabpv2_buoys.csv?"
        f"&time%3E={start_date}T00%3A00%3A00Z"
        f"&time%3C={end_date}T16%3A33%3A01Z"
    )

    print(
        "Downloading CSV file using date range: "
        f"{start_date} to {end_date}..."
        )
    
    try:
        # Send a GET request to download the file with a context manager
        with requests.get(url, headers=headers, stream=True) as response:
            if response.status_code == 200:
                # Get the total file size from the 'Content-Length' header
                total_size = int(response.headers.get('Content-Length', 0))

                with open(local_file_path, 'wb') as f:
                    with tqdm(
                            total=total_size, unit='B',
                            unit_scale=True, desc='Downloading'
                            ) as pbar:
                        # Download in chunks
                        for chunk in response.iter_content(chunk_size=8192):  
                            f.write(chunk)
                            # Update the progress bar with the size
                            # of the downloaded chunk
                            pbar.update(len(chunk))  
            else:
                raise ValueError(
                    f"Unable to download file from {url}. "
                    f"Error code: {response.status_code}"
                    )

    except requests.RequestException as e:
        raise ValueError(
            f"Unable to download file from {url}. "
            f"Error code: {e}"
            )

        
def load_ERDDAP_buoy_CSV(user_args):
    """
    Load buoy observation data from a local or newly downloaded CSV file
    for a given date range.

    This function ensures the date range is valid, downloads the ERDDAP
    buoy dataset if needed, and loads it into a pandas DataFrame. It expects
    data to be available from 2011 onwards.

    Args:
        user_args (dict): Dictionary containing script arguments with keys:
            - 'start_date' (str): Start date in 'YYYYMMDD' format.
            - 'end_date' (str): End date in 'YYYYMMDD' format.
            - 'infile_rootname' (str): Root name for the downloaded file.
            - 'work_path' (str): Path to the local directory where the
              CSV is stored.

    Returns:
        pandas.DataFrame: DataFrame containing buoy observations from the
        specified date range.
        Includes the following columns:
            - time
            - latitude
            - longitude
            - surface_temp
            - air_temp
            - bp (atmospheric pressure)
            - buoy_id
            - buoy_type
            - buoy_owner
            - logistics
            - hour
            - day_of_year

    Raises:
        ValueError: If the start date is earlier than 2011 or
                    after the end date.

    Side Effects:
        - Triggers a call to `download_erddap_buoy_csv()` to download data if
          the file is not found locally.
        - Prints status messages to the console during loading.

    Example:
        >>> df = load_ERDDAP_buoy_CSV(user_args)
        >>> df.head()
    """

    import os
    import pandas as pd
    from datetime import datetime
    

    # Ensure dates make sense    
    start_date = datetime.strptime(f"{user_args['start_date']}", "%Y%m%d")    
    end_date = datetime.strptime(f"{user_args['end_date']}", "%Y%m%d")
    if start_date.year < 2011:
        raise ValueError(
            'ERDDAP only has daily buoy observations from year 2011'
            ' to current'
            )
    if start_date > end_date:
        raise ValueError("`start_date` must not be later than `end_date`.")

    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    # Function that downloads the data and checks local CSV files
    download_erddap_buoy_csv(user_args)
    
    
    # Create data frame for dates from start_date to end_date
    print('Loading CSV data...')
    print(start_date)
    local_file_path = os.path.join(
        user_args['work_path'],
        f"{user_args['infile_rootname']}_{start_date}_{end_date}.csv"
        )

    df = pd.read_csv(
        local_file_path, low_memory=False, 
        usecols=[
            "time", "latitude", "longitude", "surface_temp", "air_temp", "bp",
            "buoy_id", "buoy_type", "buoy_owner", "logistics",
            "hour", "day_of_year"
        ]
    )
    
    return df        


def extract_daily_positions(df):
    """
    Extract daily position summaries for a single buoy from its full
    observation records.

    This function:
    - Groups the DataFrame by calendar day (based on `formatted_date`).
    - Extracts the first and last latitude/longitude and time entries
      for each day.
    - Computes the duration between the first and last observation of each day.
    - Flags days as valid if the observation span exceeds 20 hours.

    Args:
        df (pandas.DataFrame): DataFrame containing at least the following
           columns:
            - 'formatted_date' (datetime or str): Date of observation.
            - 'formatted_time' (time): Time of observation.
            - 'latitude' (float): Latitude values.
            - 'longitude' (float): Longitude values.
            - 'day_of_year' (int): Used to filter out invalid records
              (must be > 0).

    Returns:
        tuple:
            - date_range (numpy.ndarray): Array of unique dates
              (as datetime objects).
            - lat_daily (pandas.DataFrame): DataFrame with `first` and `last`
              latitude values per day.
            - lon_daily (pandas.DataFrame): DataFrame with `first` and `last`
              longitude values per day.
            - hour_daily (pandas.DataFrame): DataFrame with:
                - `duration`: Time delta between first and last observation
                  of the day.
                - `valid`: Boolean indicating if duration exceeds 20 hours.

    Notes:
        - Only includes rows where `day_of_year > 0`.
        - Assumes input data corresponds to a single buoy.
        - Time comparison is done using 24-hour clock;
          durations over midnight are not supported.

    Example:
        >>> extract_daily_positions(buoy_df)
        (array([datetime.date(2025, 1, 1), ...]),
         lat_daily_df,
         lon_daily_df,
         hour_daily_df)
    """
    
    import pandas as pd
    

        
    # create datetime field using year & DOY of the transmitted position
    # only read observations where `day_of_year` is greater than 0
    df = df[df['day_of_year'] > 0]
    df['day'] = pd.to_datetime(df['formatted_date'])
    grouped = df.groupby('day')

    # Group and aggregate
    result = grouped.agg(
        first_latitude=('latitude', 'first'),
        last_latitude=('latitude', 'last'),
        first_longitude=('longitude', 'first'),
        last_longitude=('longitude', 'last'),
        first_time=('formatted_time', 'first'),
        last_time=('formatted_time', 'last')
    ).reset_index()

    # Calculate duration
    result['duration'] = (
        pd.to_datetime(result['last_time'], format='%H:%M:%S') - 
        pd.to_datetime(result['first_time'], format='%H:%M:%S')
    )
    result['valid'] = result['duration'] > pd.Timedelta(hours=20)

    # Create data frames
    date_range = result['day'].unique()
    lat_daily = result[['first_latitude', 'last_latitude']].rename(columns={
        'first_latitude': 'first', 'last_latitude': 'last'
    })
    lon_daily = result[['first_longitude', 'last_longitude']].rename(columns={
        'first_longitude': 'first', 'last_longitude': 'last'
    })
    hour_daily = result[['duration', 'valid']]        

    
    return date_range, lat_daily, lon_daily, hour_daily


def calculate_drift_daily(lat_daily, lon_daily):
    """
    Compute daily drift metrics between the first and last recorded positions.

    This function calculates:
    - Cartesian displacements (dx, dy) in longitude and latitude.
    - Forward azimuth (bearing from start to end point).
    - Great-circle distances between the first and last positions of each day.

    Args:
        lat_daily (pandas.DataFrame): DataFrame with two columns:
            - 'first': Starting latitude for each day.
            - 'last': Ending latitude for each day.
        lon_daily (pandas.DataFrame): DataFrame with two columns:
            - 'first': Starting longitude for each day.
            - 'last': Ending longitude for each day.

    Returns:
        tuple:
            - x_first (np.ndarray): First recorded longitudes per day.
            - y_first (np.ndarray): First recorded latitudes per day.
            - dx (np.ndarray): Daily change in longitude (last - first).
            - dy (np.ndarray): Daily change in latitude (last - first).
            - fwd_azimuth (np.ndarray): Forward azimuth in degrees from
              true north.
            - total_distance (np.ndarray): Great-circle distances in meters.

    Notes:
        - NaN values in coordinates are replaced with 0.0 using
          `np.nan_to_num`.
        - Uses the `compute_bearing` function for spherical distance and
          azimuth calculations.
        - Assumes daily data with consistent 'first' and 'last' structure in
          both input DataFrames.

    Example:
        >>> x, y, dx, dy, azimuth, dist = calculate_drift_daily(lat_df, lon_df)
    """

    import numpy as np
    

    # Batch transformation for efficiency
    first_coords = np.column_stack(
        (lon_daily['first'].values, lat_daily['first'].values)
        )
    last_coords = np.column_stack(
        (lon_daily['last'].values, lat_daily['last'].values)
        )
    x_first, y_first = first_coords[:, 0], first_coords[:, 1]
    x_last, y_last = last_coords[:, 0], last_coords[:, 1]

    # Clean up invalid values in coordinates after transformation
    # Handle invalid values in a vectorized way
    x_first = np.nan_to_num(x_first, nan=0.0)
    y_first = np.nan_to_num(y_first, nan=0.0)
    x_last = np.nan_to_num(x_last, nan=0.0)
    y_last = np.nan_to_num(y_last, nan=0.0)

    # Vectorized calculation of differences
    dx, dy = x_last - x_first, y_last - y_first
    
    # Calculate distance and bearing    
    fwd_azimuth, total_distance = compute_bearing(
        lat_daily['first'].values,
        lon_daily['first'].values,
        lat_daily['last'].values,
        lon_daily['last'].values
    )
    
    return x_first, y_first, dx, dy, fwd_azimuth, total_distance


def get_avg(df, col):
    """
    Compute the daily average of a specified column in a DataFrame.

    This function groups data by the `formatted_date` column and computes the
    mean of the given column for each date. It preserves the original ordering
    and count of dates, returning `NaN` for days with no valid data.

    Args:
        df (pandas.DataFrame): Input DataFrame containing a `formatted_date`
           column and the target numeric column specified by `col`.
        col (str): The name of the column for which to compute daily averages.

    Returns:
        numpy.ndarray: A 1D array of daily average values corresponding to each 
        unique date in `df['formatted_date']`. If all values in the column are 
        `NaN`, the array will contain `NaN` values of the same length as the 
        number of unique dates.

    Notes:
        - Rows with `NaN` in the target column are excluded from averaging.
        - The output array maintains alignment with the original set of
          unique dates (in order of appearance), even if some dates have
          no valid data.

    Example:
        >>> get_avg(df, 'air_temp')
        array([272.5, 271.3, nan, 270.8])
    """    
    
    import numpy as np
    
    unique_dates = df['formatted_date'].unique()
    
    # Drop rows where col is NaN
    df_clean = df.dropna(subset=[col])
    
    if df_clean.empty:
        # All values in col are NaN — return NaNs for each unique date
        return np.nan * len(unique_dates)
    
    # Otherwise compute the mean per day
    daily_avg = df_clean.groupby('formatted_date')[col].mean()
    
    # Reindex to ensure all original dates are preserved,
    # even if missing from df_clean
    daily_avg = daily_avg.reindex(unique_dates)
    
    return daily_avg.values