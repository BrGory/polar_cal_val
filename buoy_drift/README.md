
# Polar Cal/Val Project

## Overview
This repository is part of the **Polar Cal/Val Project** aimed at processing and converting buoy data obtained from ERDDAP into GeoJSON and Dart files. The data processing includes:
- Data fetching from ERDDAP.
- Data cleaning and formatting.
- Buoy drift calculation.
- Generation of Dart files and GeoJSON outputs for use with MapBox Vector Tile Encoder/Decoder.

## Files
### 1. `helper.py`
This file contains utility functions for:
- **Fetching IABP Acronyms** from the official IABP Acronyms webpage.
- **Computing bearings and distances** between two geographical points using the WGS84 ellipsoid.
- **Downloading and loading buoy data** from ERDDAP servers.
- **Extracting daily buoy positions and calculating drift distance.**

### 2. `polar_calval_buoy_drift.py`
This file is responsible for:
- Parsing user arguments and setting configurations.
- Cleaning CSV files to retain Arctic-specific buoy data only.
- Building Dart files with daily drift information.
- Generating GeoJSON files with buoy metadata for MapBox compatibility.

### 3. **Code Structure & Workflow**
1. User specifies input dates, file paths, and other configurations via command-line arguments.
2. Data is fetched from ERDDAP and cleaned for Arctic-specific processing.
3. Buoy positions are analyzed to generate daily drift data.
4. Output files (`.dart` and `.geojson`) are created with precision control and format consistency.

## Usage
The program can be executed via the command line with the following syntax:
```
python polar_calval_buoy_drift.py -v -i <Input_CSV_Filename> -o <Output_Directory> -p <Precision>
```
### Options:
- `-s` / `--start_date`: Start date of buoy data download (Format: YYYYMMDD).
- `-e` / `--end_date`: End date of buoy data download (Format: YYYYMMDD). Defaults to today's date.
- `-i` / `--infile_rootname`: Root name of the CSV file.
- `-o` / `--output`: Output directory path.
- `-p` / `--precision`: Precision of the coordinates (default is 3 decimal places).
- `-d` / `--decimal_digits_consistent`: Enforce equal number of decimal places.
- `-c` / `--compact`: Produce compact JSON output.
- `-v` / `--verbose`: Display additional details.

## Requirements
The code requires the following Python packages:
- `requests`
- `BeautifulSoup4`
- `pyproj`
- `pandas`
- `numpy`
- `geojson`
- `geojson_rewind`
- `tqdm`

Install them via:
```
pip install -r requirements.txt
```

## License
This project is licensed under the MIT License. See the license header in the files for more information.

## Contact Information
Author: Brendon Gory (brendon.gory@noaa.gov, brendon.gory@colostate.edu)  
Supervisor: Dr. Prasanjit Dash (prasanjit.dash@noaa.gov, prasanjit.dash@colostate.edu)
