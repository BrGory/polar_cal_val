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
    Fetch and parse acronyms from the IABP Acronyms webpage.

    Parameters:
        url (str): URL of the IABP Acronyms webpage.
    Returns:
        acronym_data (dict): Acronym categories and their full name.
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
    To download CSV file within date range of `start_date` to current date
    Parameters:
        user_args (dict): Dictionary containing script arguments
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
    Load local and/or downloaded CSV file within date range of `start_date` to
    current date.
    Parameters:
        user_args (dict): Dictionary containing script arguments
    Returns:
        df (DataFrame): Data frame of all buoy observations from ERDDAP
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
    To extract coordinates (lon, lat) of a given buoy at the closest time
    DOY.000 and save to new file with daily fields.
    Parameters:
        df (DataFrame): Data frame with dates, longitudes and latitudes
    Returns:
        date_range (DataFrame): Observation dates
        lat_daily (DataFrame): Observation latitude positions
        lon_daily (DataFrame): Observation longitude positions
        hour_daily (DataFrame): Observation hours
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
    To calculate the daily drift
    (distance between first and last record of a given day).
 Parameters:
        lat_daily (DataFrame): DataFrame with columns 'first' and 'last' 
                               for daily latitude values.
        lon_daily (DataFrame): DataFrame with columns 'first' and 'last'
                               for daily longitude values.
    Returns:
        x_first (ndarray): Array of first recorded longitudes for each day.
        y_first (ndarray): Array of first recorded latitudes for each day.
        dx (ndarray): Daily change in longitude (x-direction) [last - first].
        dy (ndarray): Daily change in latitude (y-direction) [last - first].
        fwd_azimuth (ndarray): Array of forward azimuths
                               (degrees from true north, clockwise).
        total_distance (ndarray): Great-circle distances between first and 
                                  last positions per day (in meters).
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
    import numpy as np
    
    unique_dates = df['formatted_date'].unique()
    
    # Drop rows where col is NaN
    df_clean = df.dropna(subset=[col])
    
    if df_clean.empty:
        # All values in col are NaN — return NaNs for each unique date
        return np.nan * len(unique_dates)
    
    # Otherwise compute the mean per day
    daily_avg = df_clean.groupby('formatted_date')[col].mean()
    
    # Reindex to ensure all original dates are preserved, even if missing from df_clean
    daily_avg = daily_avg.reindex(unique_dates)
    
    return daily_avg.values