import pandas as pd
import pyproj as proj
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import joblib
from scipy.interpolate import CubicSpline
import sqlalchemy
import mysql.connector
import streamlit as st


# Function for adding columns with the GPS coordinates projected to a normalized x-y system
def add_norm_coord(SM_RAWfile):
    # setup the projections
    crs_wgs = proj.Proj(init='epsg:4326')  # assuming using WGS84 geographic
    crs_bng = proj.Proj(init='epsg:27700')  # use locally appropriate projected CRS

    # Convert long lat to x y
    xlong, ylat = proj.transform(crs_wgs, crs_bng, SM_RAWfile.longitude, SM_RAWfile.latitude)

    # normalize to 0-1
    xnorm = (xlong - xlong.min()) / (xlong.max() - xlong.min())
    ynorm = (ylat - ylat.min()) / (ylat.max() - ylat.min())

    # Write to dataframe
    SM_RAWfile['x'] = xnorm
    SM_RAWfile['y'] = ynorm

    # Return the new dataframe
    return SM_RAWfile


def segment_file(SM_RAWfile, SM_Sorted=0):
    # Read the raw SM file
    SM = SM_RAWfile  # Raw File

    # Add normalized coordinates in x-y system
    SM = add_norm_coord(SM)

    # Reshape values
    xcoor = SM['x'].values
    xcoor = xcoor.reshape((-1, 1))
    ycoor = SM['y'].values

    if type(SM_Sorted) == pd.core.frame.DataFrame:
        # Read corresponding T/F file identifying straight lines
        df = SM_Sorted  # Corresponding 10s T/F file

        # Sort values chronologically, reset index and isolate relevant columns
        df = df.sort_values('Start')
        df = df.reset_index()
        df = df[['Start', 'Stop', 'Eval']]

        # Set the window length and number of segments
        windowlength = 10
        rangenmb = len(df)

        # Empty variables for storing segments
        X = []
        y = []

        # For loop determining and saving relevant variables for each segment, coupling them with the evaluation
        for i in range(rangenmb):
            # Cut snippet
            start = i * windowlength
            end = windowlength * i + windowlength
            Temp = SM.iloc[start:end]

            Heading_std = Temp['HDT - Heading True'].std()
            SOG_avg = Temp['SOG - Speed over Ground'].mean()
            SOG_std = Temp['SOG - Speed over Ground'].std()
            COG_std = Temp['COG - Course over Ground'].std()
            Heel_avg = Temp['Heel'].mean()
            Heel_std = Temp['Heel'].std()
            Trim_avg = Temp['Trim Fore / Aft'].mean()
            Trim_std = Temp['Trim Fore / Aft'].std()

            reg = LinearRegression().fit(xcoor[start:end], ycoor[start:end])
            r2 = reg.score(xcoor[start:end], ycoor[start:end])

            X.append([r2, Heading_std, SOG_avg, SOG_std, COG_std, Heel_avg, Heel_std, Trim_avg, Trim_std, start, end])
            y.append([df.Eval.iloc[i]])

        X = pd.DataFrame(X, columns=['r2', 'Heading_std', 'SOG_avg', 'SOG_std', 'COG_std',
                                     'Heel_avg', 'Heel_std', 'Trim_avg', 'Trim_std', 'StartID', 'StopID'])
        y = pd.DataFrame(y, columns=['Eval'])
        SM_seg = pd.concat([X, y], axis=1)
        return SM_seg

    else:
        # Set the window length and number of segments
        windowlength = 10
        rangenmb = int(np.floor(len(SM) / windowlength))

        # Empty variables for storing segments
        X = []

        # For loop determining and saving relevant variables for each segment
        for i in range(rangenmb):
            # Cut snippet
            start = i * windowlength
            end = windowlength * i + windowlength
            Temp = SM.iloc[start:end]

            Heading_std = Temp['HDT - Heading True'].std()
            SOG_avg = Temp['SOG - Speed over Ground'].mean()
            SOG_std = Temp['SOG - Speed over Ground'].std()
            COG_std = Temp['COG - Course over Ground'].std()
            Heel_avg = Temp['Heel'].mean()
            Heel_std = Temp['Heel'].std()
            Trim_avg = Temp['Trim Fore / Aft'].mean()
            Trim_std = Temp['Trim Fore / Aft'].std()

            reg = LinearRegression().fit(xcoor[start:end], ycoor[start:end])
            r2 = reg.score(xcoor[start:end], ycoor[start:end])

            X.append([r2, Heading_std, SOG_avg, SOG_std, COG_std, Heel_avg, Heel_std, Trim_avg, Trim_std, start, end])

        SM_seg = pd.DataFrame(X, columns=['r2', 'Heading_std', 'SOG_avg', 'SOG_std', 'COG_std',
                                          'Heel_avg', 'Heel_std', 'Trim_avg', 'Trim_std', 'StartID', 'StopID'])
        return SM_seg


def plot_line_results(SM_RAWfile, Valuefile=None):
    if Valuefile is None:
        # Make a gps plot of the file
        GPSplot = px.scatter_mapbox(SM_RAWfile, lat="latitude", lon="longitude")
        GPSplot.update_layout(mapbox_style="carto-positron", mapbox_zoom=12, margin={"r": 0, "t": 0, "l": 0, "b": 0})
        GPSplot.show()
    else:
        identifier = Valuefile

        # Identify the indexes for the rows which are evaluated as straight
        # Check if resultsfile or evaluation file
        if len(identifier.columns) > 5:
            rows_to_keep = identifier.loc[identifier['Pred'] == True, ['StartID', 'StopID']]
            rows_to_keep = list(rows_to_keep.itertuples(index=False, name=None))

        elif len(identifier.columns) <= 5:
            rows_to_keep = identifier.loc[identifier['Eval'] == True, ['Start', 'Stop']]
            rows_to_keep = list(rows_to_keep.itertuples(index=False, name=None))

        # Empty variable for storing the rows with straight lines
        selected_rows = []

        # Store the rows with straight lines
        for start, stop in rows_to_keep:
            selected_rows.append(SM_RAWfile.iloc[start:stop])

        # Make a new dataframe consisting of only the straight sections
        SM_straight = pd.concat(selected_rows)

        # Add identifier to the sets
        SM_RAWfile['SET'] = np.zeros(len(SM_RAWfile), dtype=int)
        SM_straight['SET'] = np.ones(len(SM_straight), dtype=int)

        # Combine the sets for the plot
        Comb = pd.concat([SM_RAWfile, SM_straight], keys=['s1', 's2'])

        # Make a gps plot of the file with the straight parts highlighted
        GPSplot = px.scatter_mapbox(Comb, lat="latitude", lon="longitude", color="SET")
        GPSplot.update_layout(mapbox_style="carto-positron", mapbox_zoom=12, margin={"r": 0, "t": 0, "l": 0, "b": 0})
        GPSplot.show()


def identify_lines(SM_RAWfile, Predictionmodel):
    SM = SM_RAWfile

    SM_seg = segment_file(SM)

    # load the trained model from a file
    loaded_model = Predictionmodel

    # use the loaded model to make predictions
    y_pred = loaded_model.predict(SM_seg.loc[:, 'r2':'Trim_std'])

    # Add the predictions to a Result dataframe
    y = pd.DataFrame(y_pred, columns=['Pred'])
    Results = pd.concat([SM_seg, y], axis=1)
    return Results


def rolling_identify_lines(SM_RAWfile, Predictionmodel):
    SM = SM_RAWfile

    # load the trained model from a file
    loaded_model = Predictionmodel

    Dropped = SM.copy()

    windowsize = 10

    # read the original dataframe
    df_original = SM.copy()

    for i in range(windowsize):
        SM_seg = segment_file(Dropped)
        SM_seg.StartID = SM_seg.StartID + i
        SM_seg.StopID = SM_seg.StopID + i

        # use the loaded model to make predictions
        y_pred = loaded_model.predict(SM_seg.loc[:, 'r2':'Trim_std'])

        y = pd.DataFrame(y_pred, columns=['Pred'])

        df_10seg = pd.concat([SM_seg.loc[:, ['StartID', 'StopID']], y], axis=1)

        # create a dictionary with the segment start and stop indices as keys
        # and the segment true/false values as values
        seg_dict = dict(zip(zip(df_10seg['StartID'], df_10seg['StopID']), df_10seg['Pred']))

        # define a lambda function to check if the row index falls within any of the segments
        # and return the corresponding true/false value
        get_pred = lambda x: seg_dict.get(
            next(((start, stop) for start, stop in seg_dict.keys() if start <= x <= stop), None), False)

        # add a new column to the original dataframe with the corresponding true/false values
        columnname = 'Pred' + str(i)
        df_original[columnname] = df_original.index.map(get_pred)

    # df_original.to_csv('10runs.csv', index=False)

    reslt = df_original.loc[:, 'Pred0':'Pred9']

    threshold = 0.9

    reslt['Pred'] = reslt.apply(lambda row: row.mean() > threshold, axis=1)

    Results = pd.concat([SM, reslt['Pred']], axis=1)

    return Results


def plot_rolling_line_results(SM_Line_Result_file):
    # Make a gps plot of the file with the straight parts highlighted
    GPSplot = px.scatter_mapbox(SM_Line_Result_file, lat="latitude", lon="longitude", color="vmg")
    GPSplot.update_layout(mapbox_style="carto-positron", mapbox_zoom=12, margin={"r": 0, "t": 0, "l": 0, "b": 0})
    GPSplot.show()


def join_rolling_lines(SM_Line_Result_file, length=20, vmg=4):
    SM = SM_Line_Result_file

    # create a boolean mask for the True values in the "Pred" column
    mask = SM['Pred'] == True

    SM['stream_id'] = (mask != mask.shift(1)).cumsum()

    SM = SM.loc[SM['Pred'] == True]

    # identify streams that are shorter than 40 rows
    short_streams = SM['stream_id'].value_counts()[lambda x: x < length].index

    # drop all rows belonging to short streams
    SM = SM.loc[~SM['stream_id'].isin(short_streams)]

    # filter out rows where the mean of 'vmg' is less than 4
    group_means = SM.groupby('stream_id')['vmg'].mean()
    good_stream_ids = group_means.index[abs(group_means) >= vmg]
    SM = SM.loc[SM['stream_id'].isin(good_stream_ids)]

    return SM


def SM_WIND_pair(SM_Rawfile, WIND_Rawfile):
    df = WIND_Rawfile
    SM = SM_Rawfile

    # Add Datetime objects
    SM['time'] = pd.to_datetime(SM['time'])
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Drop NAN entries for wind
    df = df[df['Apparent_Wind_Angle'].notna()]
    df = df.dropna(axis=1, how='all')
    df = df.reset_index(drop=True)
    df = df.dropna()

    def calculate_true_wind(AWS, AWA, SOG, COG):
        # Convert AWA to 0-180
        AWA = np.where(AWA > 180, AWA - 360, AWA)
        # Convert speed to knots
        AWS = AWS / 1.852000
        SOG = SOG / 1.852000
        # Calculate AWD
        AWD = COG + AWA
        # Convert angles to radians
        AWD_rad = np.radians(AWD)
        COG_rad = np.radians(COG)

        # Calculate the wind components
        u = SOG * np.sin(COG_rad) - AWS * np.sin(AWD_rad)
        v = SOG * np.cos(COG_rad) - AWS * np.cos(AWD_rad)

        # Calculate the true wind speed
        TWS = np.sqrt(v ** 2 + u ** 2)

        # Calculate the TWD (global wind angle)
        TWD_rad = np.arctan2(u, v)
        TWD_deg = np.degrees(TWD_rad)

        # Normalize the angle to be between 0 and 360 degrees
        TWD_deg = (TWD_deg + 360) % 360

        return TWS, TWD_deg

    def calculate_vmg(TWD_deg, SOG, COG_deg):
        # Calculate the wind angle relative to the course over ground
        TWD_deg = TWD_deg % 360
        COG_deg = COG_deg % 360

        # Calculate the angle between the TWD and COG vectors
        angle_deg = abs(TWD_deg - COG_deg) - 180
        if angle_deg > 180:
            angle_deg = 360 - angle_deg
        if angle_deg > 0:
            angle_deg = -angle_deg

        if COG_deg > 180:
            if COG_deg >= TWD_deg >= COG_deg - 180:
                # tack = 'starboard'
                TWA_deg = -angle_deg
            else:
                # tack = 'port'
                TWA_deg = angle_deg
        else:
            if COG_deg <= TWD_deg <= COG_deg + 180:
                # tack = 'port'
                TWA_deg = angle_deg
            else:
                # tack = 'starboard'
                TWA_deg = -angle_deg

        # Convert angles to radians
        TWA_rad = np.radians(TWA_deg)

        # Calculate the component of the boat's speed in the direction of the course over ground
        vmg = SOG * np.cos(TWA_rad)

        return vmg, TWA_deg

    # Apply the Wind calculation function to the Wind dataframe
    df_apl = df.apply(
        lambda row: calculate_true_wind(
            row['Apparent_Wind_Speed(kph)'],
            row['Apparent_Wind_Angle'],
            row['Speed_Over_Ground(kph)'],
            row['Course_Over_Ground']),
        axis=1,
        result_type='expand'
    )

    # Add columns to wind dataframe
    df_apl.columns = ['TWS', 'TWA']
    df = pd.concat([df, df_apl], axis=1)

    # Filter interpolation to take place where data is valid
    index_bool = SM.time < df.time.iloc[-1]
    index_bool2 = SM.time > df.time.iloc[0]

    index_bool = index_bool & index_bool2


    # Interpolate True Wind Speed and True Wind Angle to fit SM samplingrate
    xx = pd.DatetimeIndex(SM.time[index_bool]).asi8 / (10 ** 6)

    # df = df.dropna()
    x = df.loc[:, 'timestamp']
    y = df.loc[:, ['TWS', 'TWA']]
    y['TWA'] = y['TWA'] + 360
    cub_spl = CubicSpline(x, y, extrapolate=None)

    yy = cub_spl(xx)

    # Add results to dataframe and normalize to 0 - 360 heading
    WindData = pd.DataFrame(data=yy, columns=['WindSpeed', 'WindDirection'])
    WindData['WindDirection'] = abs((WindData['WindDirection'] - 360) % 360)

    # Concatenate results to SM file
    SM = pd.concat([SM, WindData], axis=1)

    # Calculate VMG
    SM_apl = SM.apply(
        lambda row: calculate_vmg(
            row['WindDirection'],
            row['SOG - Speed over Ground'],
            row['COG - Course over Ground']),
        axis=1,
        result_type='expand'
    )

    # Add columns to SM dataframe
    SM_apl.columns = ['vmg', 'TWA']
    SM = pd.concat([SM, SM_apl], axis=1)
    SM['vmg'] = abs(SM['vmg'])
    return SM


def upload_session(SM_Rawfile, Wind_Rawfile, Log, Predictionmodel, database_username='root', database_password='49er',
                   database_ip='127.0.0.1', database_name='bac'):
    # If exists behavior: fail/append/replace
    ifexist = 'append'

    # Load data
    LOG = Log

    SM = SM_Rawfile
    SM = SM.loc[:, ['time', 'servertime', 'latitude', 'longitude', 'HDT - Heading True',
                    'SOG - Speed over Ground', 'COG - Course over Ground', 'Heel',
                    'Trim Fore / Aft', 'GPS Satellites', 'GPS HAcc']]

    WIND = Wind_Rawfile

    print('Data loaded')

    # Pair wind and SM data
    SM = SM_WIND_pair(SM, WIND)

    print('Data Paired')

    # load the trained model from a file
    loaded_model = Predictionmodel

    print('Model Loaded')

    # Identify segments
    SM = rolling_identify_lines(SM, loaded_model)

    print('Segments identified')

    # Join and filter segments
    SM = join_rolling_lines(SM)

    print('Segments filtered and joined')

    # establish a connection to the MySQL server for SQLAlchemy
    database_connection = sqlalchemy.create_engine('mysql+mysqlconnector://{0}:{1}@{2}/{3}'.
                                                   format(database_username, database_password,
                                                          database_ip, database_name))

    print('Connection with sqlAlch made')

    # establish a connection to the MySQL server for mysql.connecter
    # connect to the database
    cnx = mysql.connector.connect(user='root', password='49er',
                                  host='127.0.0.1', database='bac')
    # create a cursor object
    cursor = cnx.cursor()

    print('Connection with mysql made')

    # Increment session nr.
    # execute the query
    query = ("SELECT MAX(session_id) FROM saildata")
    cursor.execute(query)
    # fetch the results
    results = cursor.fetchall()
    SM['session_id'] = results[0][0] + 1

    print(SM.sample(10).to_string())

    # SM['session_id'] = 1

    print('ID incrementet')

    # close the cursor and connection
    cursor.close()
    cnx.close()
    print('connection closed')

    # Grupér runs
    runs = SM.groupby(['stream_id'])

    # Lav run overview
    run_overview = pd.DataFrame(
        [SM.stream_id.unique(), runs.latitude.count(), runs['SOG - Speed over Ground'].mean(), runs['vmg'].mean(),
         runs['WindSpeed'].mean()], ['stream_id', 'run_length', 'avg_SOG', 'avg_VMG', 'avg_WindSpeed'])
    run_overview = run_overview.transpose()
    run_overview['session_id'] = SM['session_id'].iloc[1]

    print('Run overview made')

    # Lav session overview
    session_overview = LOG.drop(columns=['Log'], axis=1)

    delta = SM.time.iloc[-1] - SM.time.iloc[0]

    session_overview['session_id'] = SM['session_id'].iloc[1]
    session_overview['Date'] = SM.time.iloc[0].date()
    session_overview['session_length'] = delta.total_seconds()
    session_overview['Median_Wind'] = SM.WindSpeed.median()

    print('Session overview made')

    SM = SM.rename(columns={'HDT - Heading True': 'Yaw'})
    SM = SM.rename(columns={'Heel': 'Roll'})
    SM = SM.rename(columns={'Trim Fore / Aft': 'Pitch'})
    SM = SM.rename(columns={'SOG - Speed over Ground': 'SOG'})
    SM = SM.rename(columns={'COG - Course over Ground': 'COG'})
    SM = SM.drop('servertime', axis=1)

    print('Saildata updated')

    # Import saildata into MySQL
    SM.to_sql(con=database_connection, name='saildata', if_exists=ifexist)
    print('Saildata imported')

    # Import run_overview into MySQL
    run_overview.to_sql(con=database_connection, name='run_overview', if_exists=ifexist)
    print('run_overview imported')

    # Import session_overview into MySQL
    session_overview.to_sql(con=database_connection, name='session_overview', if_exists=ifexist)
    print('session_overview imported')

    print('Done')

@st.cache_data
def upload_session_step1(SM_Rawfile, Wind_Rawfile, _SegmentPredictionmodel, vmgmin=4, lenghtmin=20):

    SM = SM_Rawfile
    SM = SM.loc[:, ['time', 'servertime', 'latitude', 'longitude', 'HDT - Heading True',
                    'SOG - Speed over Ground', 'COG - Course over Ground', 'Heel',
                    'Trim Fore / Aft', 'GPS Satellites', 'GPS HAcc']]

    WIND = Wind_Rawfile

    print('Data loaded')

    # Pair wind and SM data
    SM = SM_WIND_pair(SM, WIND)

    print('Data Paired')

    # load the trained model from a file
    loaded_model = _SegmentPredictionmodel

    print('Model Loaded')

    # Identify segments
    SM = rolling_identify_lines(SM, loaded_model)

    print('Segments identified')

    return SM


def fetch_crew():
    # establish a connection to the MySQL server for mysql.connecter
    # connect to the database
    cnx = mysql.connector.connect(user='root', password='49er',
                                  host='127.0.0.1', database='bac')
    # create a cursor object
    cursor = cnx.cursor()

    # execute the query
    query = ("SELECT DISTINCT Crew FROM session_overview")
    cursor.execute(query)
    # fetch the results
    results = cursor.fetchall()

    list_of_crew = [x[0] for x in results]

    # close the cursor and connection
    cursor.close()
    cnx.close()

    return list_of_crew
