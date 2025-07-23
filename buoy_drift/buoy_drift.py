"""
******************************************************************************

 Project:    Polar Cal/Val
 Purpose:    Convert buoy CSV file downloaded from ERDDAP to GeoJSON and 
             .dart files
             - check for precision
             - check for consistency in number of digits after the decimal
               point. This is mathematically irrelevant but technically
               critical for the Mapbox Vector Tile Encoder/Decoder to 
               work properly.
               e.g., lat: 30.2567   30.2 are inconsistent;
               change to 30.2567  30.1999
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

  Call syntax: buoy_drift.py -v for verbose
               buoy_drift.py -h for help
  buoy_drift.py -v -s 20210101 -p 3

"""


def read_arguments():
    """
    Parse and validate command-line arguments for converting buoy CSV files 
    to GeoJSON and Dart formats.
    
    This function sets up an `argparse.ArgumentParser` to collect the
    following arguments:
    
    - `--start_date` (`-s`): Required start date for buoy data download
      (format: YYYYMMDD).
    - `--end_date` (`-e`): Optional end date for data download
      (default: today).
    - `--infile_rootname` (`-i`): Optional name prefix for the input CSV file 
      (default: "ERDDAP_IABP").
    - `--output` (`-o`): Directory path to save downloaded CSV and output files 
      (default: predefined path on local system).
    - `--precision` (`-p`): Number of digits after the decimal point
      for coordinates (default: 3).
    - `--decimal_digits_consistent` (`-d`): Flag to enforce fixed
      decimal digits in coordinate formatting for compatibility with MapBox
      (default: False).
    - `--compact` (`-c`): Flag to make GeoJSON output compact, with 
      no whitespace or newlines (default: False).
    - `--verbose` (`-v`): Flag to print parsed configuration parameters
      (default: False).
    
    Returns:
        dict: A dictionary of user-specified or default argument
              values with the following keys:
            - 'start_date' (str)
            - 'end_date' (str)
            - 'infile_rootname' (str)
            - 'work_path' (str)
            - 'precision' (int)
            - 'decimal_digits_consistent' (bool)
            - 'compact' (bool)
            - 'verbose' (bool)
    
    Exits:
        Exits the program with status code 1 if:
            - `--start_date` is not provided.
            - Either `--start_date` or `--end_date` is not in the
              format YYYYMMDD.
    
    Side Effects:
        - Creates output directories if they don't already exist.
        - Optionally prints parsed parameters if `--verbose` is enabled.
    
    Example:
        $ python script.py -s 20250101 -e 20250115 -v
    """
    
    import argparse
    import sys
    import os
    from datetime import datetime
    
    # Get today's date in the required format
    current_date = datetime.now().strftime("%Y%m%d")

    parser = argparse.ArgumentParser(
        description='Converts buoy CSV file to GeoJSON and Dart files.'
        )
    
    parser.add_argument(
        '-s', '--start_date',
        default=None, type=str,
        action='store',
        help='Start date to download buoy data. Date must be YYYYMMDD'
        )
    
    parser.add_argument(
        '-e', '--end_date',
        default=None, type=str,
        action='store',
        help='End date to download buoy data. Date must be YYYYMMDD. '
            f'If not provided, today\'s date ({current_date}) will be used.'
        )    
    
    parser.add_argument(
        '-i', '--infile_rootname',
        default='ERDDAP_IABP',
        action='store',
        help='Root name of the downloaded CSV file'
        )
    
    parser.add_argument(
        '-o', '--output',
        default=r'D:\NOAA\Analysis\USNIC products\IABP\buoys',
        action='store',
        help='Dir on file sys to store downloaded CSV file and'
             'create output files',        
        )
    
    parser.add_argument(
        '-p', '--precision',
        default=3, type=int,
        action='store',
        help="digits after decimal point"
        )
    
    parser.add_argument(
        "-d", "--decimal_digits_consistent",
        default=False,
        action="store_true",
        help="Enforce the number of digits after decimal in coordinate"
             "values to be of equal length to align with the MapBox "
             "Encoder/Decoder. Else the encoder/decoder misbehaves. "
             "Default is false."
        )
    
    parser.add_argument(
        "-c", "--compact",
        default=False,
        action="store_true",
        help="Make output JSON compact without newline and whitespaces. "
             "Default is False"
             )
    
    parser.add_argument(
        "-v", "--verbose",
        default=False,
        action="store_true",
        help="Show configuration parameters and further details. "
             "Default is false."
        )
    
    
    # Read command line args.
    args = parser.parse_args()
    
    
    # Validate dates
    if args.start_date == None:
        print("Error: `-s` Start Date is missing.")
        sys.exit(1)
        
    start_date = args.start_date
    if args.end_date == None:
        end_date = current_date
    else:
        end_date = args.end_date

       
    try:
        datetime.strptime(start_date, "%Y%m%d")
    except ValueError:
        err_msg = (
            f"Error: Start date '{start_date}' "
            "is not in the required format YYYYMMDD."
            )
        print(err_msg)
        sys.exit(1)
        
    try:
        datetime.strptime(end_date, "%Y%m%d")
    except ValueError:
        err_msg = (
            f"Error: End date '{end_date}' "
              "is not in the required format YYYYMMDD."
              )
        print(err_msg)
        sys.exit(1)        
        

    
    # Ensure paths are available
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'dart'), exist_ok=True)
    
    
    user_args = {
        'start_date': start_date,
        'end_date': end_date,
        'infile_rootname': args.infile_rootname,
        'work_path': os.path.normpath(os.path.join(args.output)),
        'precision': args.precision,
        'decimal_digits_consistent': args.decimal_digits_consistent,
        'compact': args.compact,
        'verbose': args.verbose
    }
    
    param_string = (
        "CONF PARAMS:\n"
        " start date (-r)[0]:                    "
        f"{user_args['start_date']}\n"
        " end date (-r)[1]:                      "
        f"{user_args['end_date']}\n"
        " input csv file folder or file (-i):    "
        f"{user_args['infile_rootname']}\n"
        " output geojson folder or file (-o):    "
        f"{user_args['work_path']}\n"
        " precision (-p):                        "
        f"{user_args['precision']}\n\n"
        " Boolean args below. If key is provided then 'True', else 'False'.\n"
        " decimal digits consistent (-d) output:"
        f"{user_args['decimal_digits_consistent']}\n"
        " compact (-c) output:                   "
        f"{user_args['compact']}\n"
    )

    if user_args['verbose'] is True:
        print(param_string)
    
    
    return user_args


def subset_csv(df, user_args, arctic=True):
    """
    Clean and subset buoy data from a CSV DataFrame based 
    on hemisphere and formatting rules.
    
    This function performs the following operations:
    - Filters rows by hemisphere (Arctic or Antarctic) using latitude.
    - Converts and formats date and time columns.
    - Cleans `buoy_id` and `hour` columns by removing trailing decimals.
    - Rounds coordinate columns (`latitude`, `longitude`) to the 
      specified precision.
    - Ensures temperature and pressure fields are properly cast to float.
    
    Args:
        df (pandas.DataFrame): The raw DataFrame read from a buoy CSV file.
        user_args (dict): Dictionary of configuration parameters. Expects:
            - 'precision' (int): Number of decimal places to round coordinates.
        arctic (bool, optional): If True, subsets data for the Arctic region 
            (latitude > 0). If False, subsets for the Antarctic (latitude < 0).
            Defaults to True.
    
    Returns:
        pandas.DataFrame: Cleaned and filtered DataFrame ready for
        further processing or export.
    
    Notes:
        - Assumes the first row of the CSV (after the header) is metadata
          and skips it.
        - Adds two columns:
            - `formatted_date`: Date portion of the timestamp (YYYY-MM-DD).
            - `formatted_time`: Time portion (HH:MM:SS).
        - Drops the original `time` column.
        - Converts key columns to appropriate types for numeric and
          geographic data integrity.
    """
    
    import pandas as pd
    
    precision = user_args['precision']
    
    # Ignore first row (not the header row) in the ERDDAP csv file
    df = df.iloc[1:].reset_index(drop=True)
    
    
    if arctic:
        # Get only buoys in the Arctic region. For simplicity, latitude > 0
        df = df[df['latitude'].astype(float) > 0]
    else:
        # Get only buoys in the Antarctic region. For simplicity, latitude < 0
        df = df[df['latitude'].astype(float) < 0]
    
    # Add formatted date column as YYYY-MM-DD
    df['time'] = pd.to_datetime(df['time'])
    df['formatted_date'] = df['time'].dt.date
    df['formatted_date'] = pd.to_datetime(df['formatted_date'])
    
    
    # Add formatted time column as HH-MM-SS
    df['formatted_time'] = df['time'].dt.time
    df.drop(['time'], axis=1, inplace=True)
    
    
    # Column `buoy_id` is inferred as float64.
    # Convert to string then strip '.0'
    df['buoy_id'] = df['buoy_id'].astype(str).str.replace('.0','')
    
    
    # Strip '.0' from column `hour`
    df['hour'] = df['hour'].astype(str).str.replace('.0','')
    
    
    # Set coordinates to float
    df['longitude'] = round(df['longitude'].astype(float), precision)
    df['latitude'] = round(df['latitude'].astype(float), precision)
    
    
    # Set temperatures and pressure to float
    df['air_temp'] = df['air_temp'].astype(float)
    df['surface_temp'] = df['surface_temp'].astype(float)
    df['bp'] = df['bp'].astype(float)
    
    return df


def build_dart(df, user_args, iabp_acronyms, arctic):
    """
    Generate `.dart` metadata files for individual buoys based on daily movement 
    and environmental data.

    For each buoy in the provided DataFrame, this function:
    - Extracts daily position and calculates drift statistics 
      (e.g., dx, dy, distance, bearing).
    - Computes average environmental parameters
      (air temperature, surface temperature, pressure).
    - Builds a structured `.dart` file containing this information
      in tabular format.
    - Writes each `.dart` file to the specified output directory.
    - Collects metadata (e.g., type, owner, program, final location)
      for return.

    Args:
        df (pandas.DataFrame): Cleaned and filtered buoy data with 
            required columns,including `buoy_id`, `latitude`, `longitude`,
            `air_temp`, `surface_temp`, `bp`, `time`, `buoy_owner`,
            `buoy_type`, and `logistics`.
        user_args (dict): Dictionary of configuration parameters, expects:
            - 'precision' (int): Number of decimal places to round 
              numeric values.
            - 'work_path' (str): Output path where `.dart` files will be saved.
        iabp_acronyms (dict): Dictionary with acronym lookups for
            buoy metadata, containing:
            - 'Buoy Owner Acronyms'
            - 'Buoy Logistics Acronyms'
            - 'Buoy Type Acronyms'
        arctic (bool): Indicates whether the buoys are in the Arctic (`True`)
            or Antarctic (`False`), used for log messages only.

    Returns:
        dict: Dictionary containing metadata for each buoy keyed by `buoy_id`.
            Each value includes:
            - 'buoy_id' (str)
            - 'owner' (str)
            - 'program' (str)
            - 'buoy_type' (str)
            - 'file_name' (str)
            - 'station_info' (str)
            - 'latitude' (float): Final daily latitude
            - 'longitude' (float): Final daily longitude

    Side Effects:
        - Writes `.dart` files to the path: `<user_args['work_path']>/dart/`.
        - Prints progress using `tqdm`.

    Requires:
        - `helper.extract_daily_positions(df)`
        - `helper.calculate_drift_daily(lat, lon)`
        - `helper.get_avg(df, column_name)`

    Raises:
        - May raise `KeyError` or `IndexError` if expected columns are missing 
          or improperly formatted.

    Example:
        >>> build_dart(df, user_args, acronyms, arctic=True)
        {
            '300234063339000': {
                'buoy_id': '300234063339000',
                'owner': 'NOAA',
                'program': 'IABP',
                'buoy_type': 'SVP',
                'file_name': '300234063339000.dart',
                'station_info': 
                'https://iabp.apl.uw.edu/raw_plots.php?bid=300234063339000',
                'latitude': 84.627,
                'longitude': -133.252
            },
            ...
        }
    """
    
    import helper
    import os
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    
    buoys = {}
    buoy_ids = df['buoy_id'].unique()
    precision = user_args['precision']
    
    if arctic:
        region = 'Arctic'
    else:
        region= 'Antarctic'
        
    for buoy_id in tqdm(
            buoy_ids, desc=f'Creating dart for each {region} buoy'):
        buoy_data = {}
        
        # Create data frame for one buoy
        df_buoy = df[df['buoy_id'] == buoy_id]
        
        buoy_data['buoy_id'] = buoy_id
        
        
        # Get buoy owner        
        owner_id = df_buoy['buoy_owner'].iloc[0]
        buoy_data['owner'] = iabp_acronyms['Buoy Owner Acronyms'].get(
            owner_id, owner_id
            )
        
        # Get buoy program
        program_id = df_buoy['logistics'].iloc[0]
        buoy_data['program'] = iabp_acronyms['Buoy Logistics Acronyms'].get(
            program_id, program_id
            )
        
        # Get buoy type
        buoy_type_id = df_buoy['buoy_type'].iloc[0]    
        buoy_data['buoy_type'] = iabp_acronyms['Buoy Type Acronyms'].get(
            buoy_type_id, buoy_type_id
            )
        
        buoy_data['file_name'] = f'{buoy_id}.dart'
        buoy_data['station_info'] = (
            f'https://iabp.apl.uw.edu/raw_plots.php?bid={buoy_id}'
            )
        
        buoy_date, buoy_lat, buoy_lon, buoy_duration = (
            helper.extract_daily_positions(df=df_buoy.copy())
            )
        
        x, y, dx, dy, fwd_azimuth, distance = (
            helper.calculate_drift_daily(buoy_lat, buoy_lon)
            )
        
        distance = np.round(distance / 1000, precision)
        bearing = np.round(fwd_azimuth, precision)
        
        # print(buoy_id)
        avg_air_temp = np.round(
            helper.get_avg(df_buoy, 'air_temp'), precision
            )
        
        avg_surface_temp = np.round(
            helper.get_avg(df_buoy, 'surface_temp'), precision
            )
        
        avg_bp = np.round(
            helper.get_avg(df_buoy, 'bp'), precision
            )
        
        
        df_daily_data = pd.DataFrame({
            'date': buoy_date,
            'buoy_id': buoy_id,
            'lon': x, 'lat': y, 
            'dx': dx, 'dy': dy,
            'bearing': bearing,
            'total_distance_km': distance,
            'avg_air_temp': avg_air_temp,
            'avg_surface_temp': avg_surface_temp,
            'avg_bp': avg_bp
        })
        # Ensure there are no missing values
        df_daily_data.replace([np.inf, -np.inf], 0, inplace=True) 
        
        
        buoy_data['latitude'] = df_daily_data['lat'].iloc[-1]
        buoy_data['longitude'] = df_daily_data['lon'].iloc[-1]
        
        output_file_path = os.path.join(
            user_args['work_path'], 'dart', buoy_data['file_name']
            )
        header = (
            "#YYYY\tMM\tDD\thh\tmm\tss\tLongitude\tLatitude\t"
            "Distance\tBearing\tAir_Temperature\tSurface_Temperature\t"
            "Atmospheric_Pressure\n"
            "#year\tmo\tdy\thr\tmn\ts\tWGS84\tWGS84\tkm\tdeg\tC\tC\tmb\t\n"
        )
        with open(output_file_path, 'w') as file:
            file.write(header)

            for idx, row in df_daily_data.iterrows():
                year = str(row['date'].year)
                month = f"{row['date'].month:02}"
                day = f"{row['date'].day:02}"
                lon = f"{row['lon']}"
                lat = f"{row['lat']}"
                distance = f"{row['total_distance_km']}"
                bearing = f"{row['bearing']}"
                avg_air_temp = f"{row['avg_air_temp']}"
                avg_surface_temp = f"{row['avg_surface_temp']}"
                avg_bp = f"{row['avg_bp']}"
                data_line = (
                    f"{year}\t{month}\t{day}\t00\t00\t00\t"
                    f"{lon}\t{lat}\t"
                    f"{distance}\t{bearing}\t{avg_air_temp}\t"
                    f"{avg_surface_temp}\t{avg_bp}\n"
                    )
                file.write(data_line)
                
                
                
        buoys[buoy_id] = buoy_data
    
    return buoys


def num_of_digits_in_float(n):
    """
    Compute the number of digits in a float.

    This function returns two values:
    1. The total number of characters in the float
       (excluding the decimal point), 
       including leading digits, trailing digits, and the decimal part.
    2. The number of digits after the decimal point (i.e., precision).

    Args:
        n (float or str): A floating-point number or numeric string.

    Returns:
        list: A two-element list:
            - [0]: Total number of characters in the number string minus 1
              (exclude the decimal point).
            - [1]: Number of digits after the decimal point.

    Example:
        >>> num_of_digits_in_float(12.345)
        [5, 3]  # '12.345' → 6 chars, minus 1 = 5; 3 digits after decimal
    """    
    
    import decimal
    a = decimal.Decimal(str(n))
    return [len(str(a)) - 1, len(str(a).split('.')[1])]


def create_geojson(buoys, user_args, arctic):
    """
    Generate a GeoJSON file representing the most recent position of
    active buoys.

    For each buoy in the input dictionary, this function:
    - Extracts metadata (e.g., type, owner, program, coordinates).
    - Constructs a GeoJSON Point feature for the buoy location.
    - Ensures coordinate precision consistency if specified.
    - Assembles all features into a GeoJSON FeatureCollection.
    - Writes the result to a `.geojson` file in the output directory.

    Args:
        buoys (dict): Dictionary of buoy metadata, where each key is a
            `buoy_id` and each value is a dict with:
            - 'buoy_type' (str)
            - 'owner' (str)
            - 'program' (str)
            - 'latitude' (float)
            - 'longitude' (float)
            - 'file_name' (str): Name of associated `.dart` file
            - 'station_info' (str): URL to buoy metadata
        user_args (dict): Dictionary of user-defined parameters, expects:
            - 'precision' (int): Number of decimal digits for coordinates.
            - 'decimal_digits_consistent' (bool): Enforce equal-length decimal
               precision if True.
            - 'compact' (bool): If True, minimize file whitespace.
            - 'work_path' (str): Directory where the output file is saved.
        arctic (bool): If True, the file is named for Arctic buoys; otherwise,
            for Antarctic buoys.

    Returns:
        None

    Side Effects:
        - Writes a GeoJSON file to
         `<work_path>/arctic_buoy_dart_active.geojson` or 
          `<work_path>/antarctic_buoy_dart_active.geojson` depending on the
          `arctic` flag.
        - Applies GeoJSON winding order compliance using `geojson_rewind`.

    Notes:
        - Coordinate precision is enforced using `geojson.utils.map_coords`.
        - The decimal adjustment logic ensures consistent formatting for
          encoders like MapBox when enabled.
        - This function depends on `num_of_digits_in_float`,
          `geojson`, and `geojson_rewind`.

    Example:
        >>> create_geojson(buoys, user_args, arctic=True)
        # Creates 'arctic_buoy_dart_active.geojson' in the specified
          output folder.
    """
    
    import os
    import geojson
    import geojson_rewind
    
    # adjustment to enforce number of digits after decimal point 
    # if 'decimal_digits_consistent' key is chosen
    decimal_digit_adjustment = (10 ** (-user_args['precision']))

         
    features = []      
    for buoy_id in buoys:
        
        # Set buoy variables
        buoy = buoys[buoy_id]
        buoy_type = buoy['buoy_type']
        owner = buoy['owner']
        program = buoy['program']
        latitude = buoy['latitude']
        longitude = buoy['longitude']
        file_name = buoy['file_name']
        station_info = buoy['station_info']
        
        # geojson parts
        geometry = {
            'type': 'Point',
            'coordinates': [longitude, latitude]
            }
            
        """
        Code from csv_to_geojson.py
        Dr. Prasanjit Dash, prasanjit.dash@noaa.gov
                            prasanjit.dash@colostate.edu
        CSU CIRA Research Scientist III
        (Program Innovation Scientist)        
        2025
        """
        mod = getattr(geojson, geometry['type'])
        
        coords = geometry['coordinates']
        adjusted_coords = []

        # Enforce equal number of digits after decimal point
        if user_args['decimal_digits_consistent']:
            for x in coords:
                if num_of_digits_in_float(x)[1] < user_args['precision']:
                    x += decimal_digit_adjustment
                adjusted_coords.append(x)
        else:
            adjusted_coords = coords

        new_geometry = geojson.utils.map_coords(
            lambda x: x,  # map_coords still expects a callable
            mod(adjusted_coords, precision=user_args['precision'])
        )
        
        properties = {
                "ID": buoy_id,
                "Type": buoy_type,
                "Owner": owner,
                "Program": program,
                "Latitude": str(latitude),
                "Longitude": str(longitude),
                "Timeseries": file_name,
                "StationInfo": station_info        
            }
            
        feature = {
            "type": "Feature",
            "properties": properties,
            "geometry": new_geometry
            }        

        # compliance check
        compliant_feature_string = geojson_rewind.rewind(
            geojson.dumps(feature)
            )
        compliant_feature_object = geojson.loads(compliant_feature_string)
        features.append(compliant_feature_object)
    
    # Define the output path for the GeoJSON file
    if arctic:
        geojson_filename = 'arctic_buoy_dart_active.geojson'
    else:
        geojson_filename = 'antarctic_buoy_dart_active.geojson'
        
    output_geojson_path = os.path.join(
        user_args['work_path'], geojson_filename
        )
    
    # Create geojson header    
    feature_collection = {
        "type": "FeatureCollection",
        "features": features
        }
    
    # Write to file
    with open(output_geojson_path, "w", encoding="utf-8") as geojson_file:
        if user_args['compact']:
            geojson.dump(feature_collection, geojson_file, indent=None)
        else:
            geojson.dump(feature_collection, geojson_file, indent=1)


def main():
    """
    Main execution function for processing buoy data and generating
    output files.

    Workflow:
    1. Parses command-line arguments provided by the user.
    2. Downloads buoy CSV data from ERDDAP using helper utilities.
    3. Loads acronym mappings used to decode buoy metadata.
    4. Processes data separately for both Arctic and Antarctic regions:
        - Filters the DataFrame by hemisphere.
        - Computes drift and daily statistics to build `.dart` files.
        - Generates GeoJSON files with buoy metadata and location.
    5. Writes output files to the directory specified in `user_args`.

    Args:
        None

    Returns:
        None

    Side Effects:
        - Downloads data from ERDDAP and saves output files:
            - Dart files: `<output_dir>/dart/*.dart`
            - GeoJSON files: 
                - `arctic_buoy_dart_active.geojson`
                - `antarctic_buoy_dart_active.geojson`
        - Prints a final "done" message when complete.

    Dependencies:
        - Relies on functions and utilities from the `helper` module:
            - `load_ERDDAP_buoy_CSV`
            - `fetch_iabp_acronyms`
        - Also depends on local functions:
            - `read_arguments`
            - `subset_csv`
            - `build_dart`
            - `create_geojson`
    """
    
    import helper
    
    # parse user arguments
    user_args = read_arguments()
    
    # download buoy data from ERDDAP
    df = helper.load_ERDDAP_buoy_CSV(user_args)
    
    # define acronyms found in buoy data
    iabp_acronyms = helper.fetch_iabp_acronyms()
    
    # Create gejson and dart files for Arctic and Antarctic regions
    for arctic in [True, False]:
        df_clean = subset_csv(df, user_args, arctic)
        
        buoys = build_dart(df_clean, user_args, iabp_acronyms, arctic)
        
        create_geojson(buoys, user_args, arctic)
    
    print('done')

    
if __name__ == "__main__":
    main()

