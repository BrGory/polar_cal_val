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
            "#YYYY\tMM\tDD\thh\tmm\tss\tDistance\tBearing\t"
            "Air Temperature\tSurface Temperature\tAtmospheric Pressure\n"
            "#year\tmo\tdy\thr\tmn\ts\tkm\tdeg\tC\tC\tmb\t\n"
        )
        with open(output_file_path, 'w') as file:
            file.write(header)

            for idx, row in df_daily_data.iterrows():
                year = str(row['date'].year)
                month = f"{row['date'].month:02}"
                day = f"{row['date'].day:02}"
                distance = f"{row['total_distance_km']}"
                bearing = f"{row['bearing']}"
                avg_air_temp = f"{row['avg_air_temp']}"
                avg_surface_temp = f"{row['avg_surface_temp']}"
                avg_bp = f"{row['avg_bp']}"
                data_line = (
                    f"{year}\t{month}\t{day}\t00\t00\t00\t"
                    f"{distance}\t{bearing}\t{avg_air_temp}\t"
                    f"{avg_surface_temp}\t{avg_bp}\n"
                    )
                file.write(data_line)
                
                
                
        buoys[buoy_id] = buoy_data
    
    return buoys


# Required utility function
def num_of_digits_in_float(n):
    import decimal
    a = decimal.Decimal(str(n))
    return [len(str(a)) - 1, len(str(a).split('.')[1])]


def create_geojson(buoys, user_args, arctic):
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

