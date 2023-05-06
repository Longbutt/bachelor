import pandas as pd
from query import *
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import datetime
import SM_handling
from random import randint
from file_handling import *
import openpyxl
import joblib
import re
import json
import os
import json
import os
import subprocess
import psutil

# Set up the Google Application Credentials environment variable
gcp_service_account_key = json.loads(st.secrets["GCP"]["SERVICE_ACCOUNT_KEY"])
with open("key.json", "w") as key_file:
    json.dump(gcp_service_account_key, key_file)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"

# Start the Cloud SQL Proxy if it's not already running
proxy_process = None
for process in psutil.process_iter(["name", "cmdline"]):
    if "cloud_sql_proxy" in process.info["name"]:
        proxy_process = process
        break

if proxy_process is None:
    proxy_command = "curl -o cloud_sql_proxy https://dl.google.com/cloudsql/cloud_sql_proxy.linux.amd64 && chmod +x cloud_sql_proxy && ./cloud_sql_proxy -instances=sailingproject:europe-north1:sailproject=tcp:3306"
    proxy_process = subprocess.Popen(proxy_command, shell=True, preexec_fn=os.setsid)

# Add a st.stop() at the end of your app to prevent re-runs


def create_relative_time(group):
    first_value = group['time'].iloc[0]
    group['rel_time'] = group['time'] - first_value
    return group


# Create session, that can be refreshed when wanted
if "state" not in st.session_state:
    st.session_state.state = {}
if "widget_key" not in st.session_state.state:
    st.session_state.state["widget_key"] = str(randint(1000, 100000000))

# Header in App
st.title('Performance Analysis')

# Text in app
st.text('This is a Web App to analyse and access the performance of 49er sailors')
# Layout
st.sidebar.title('Navigation')
options = st.sidebar.radio('Pages', options=['Home', 'Analysis',
                                             'Data Statistics', 'Upload', 'Draft'])

# # Uploaded file is saved in this variable
# uploaded_files = st.sidebar.file_uploader('Upload your file here', type=['xlsx', 'csv', 'png', 'jpeg', 'jpg'],
#                                           key=st.session_state.state['widget_key'])
# # If the file has been uploaded, then perform these actions
# if uploaded_files is not None:
#     if uploaded_files.size == 0:
#         st.sidebar.warning('File is empty.')
#     file_name = uploaded_files.name
#     file_extension = file_name.split('.')[1]
#     file_index = file_name.split('.')[0]
#     if file_extension == 'csv' or 'xlsx':
#         save_uploaded_file(uploaded_files, file_extension, file_index)
#     elif file_extension == 'jpg' or 'png' or 'jpeg':
#         save_uploaded_image(uploaded_files)
#     else:
#         st.error('Invalid file type. Only .xlsx, .csv and image files are supported')

# Overview of all files in database
# Overview of all files in database
if options == 'Analysis':
    # Possible data to show
    datatype = st.radio('Plot type', options=['Graph', 'Map', 'Polar'], horizontal=True)
    # Data or plot
    plot, data = st.tabs(['Plot', 'Data'])

    # Search criteria
    wind_cond = st.radio('Wind conditions', options=['All', 'Onshore', 'Offshore'], horizontal=True)
    crew = st.radio('Crew', options=['All', 'Frederik/Jakob', 'Daniel/Buhl', 'Kristian/Marcus', 'JC/Mads'],
                    horizontal=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        current_options = ['All', '0', '1', '2']
        current_idx = st.radio('Current', options=current_options, key='current')
        current = current_options[current_options.index(current_idx)] if current_idx != 'All' else None

    with col2:
        wave_options = ['All', '0', '1', '2', '3']
        wave_idx = st.radio('Waves', options=wave_options, key='waves')
        waves = wave_options[wave_options.index(wave_idx)] if wave_idx != 'All' else None

    with col3:
        rating_options = ['All', '1', '2', '3', '4', '5']
        rating_idx = st.radio('Rating', options=rating_options, key='rating')
        rating = rating_options[rating_options.index(rating_idx)] if rating_idx != 'All' else None

    with col4:
        vmg_avg = st.number_input('Minimum VMG Average:', value=0)
        wind_avg = st.number_input('Minimum Windspeed Average', value=0)


    crew_map = {'All': None, 'Frederik/Jakob': 'Frederik/Jakob', 'Daniel/Buhl': 'Daniel/Buhl',
                'Kristian/Marcus': 'Kristian/Marcus', 'JC/Mads': 'JC/Mads'}
    wind_map = {'All': None, 'Onshore': 'Onshore', 'Offshore': 'Offshore'}
    crew_param = crew_map.get(crew)
    wind_param = wind_map.get(wind_cond)
    if crew_param or wind_param:
        available_dates = query_database('unique_dates', crew=crew_param, wind=wind_param)['DATE']
    else:
        available_dates = query_database('unique_dates')['DATE']
    dates = pd.to_datetime(available_dates).dt.strftime("%Y-%m-%d").tolist()
    date = st.selectbox('Date', ['All'] + dates, key='option_plot')
    if date != 'All':
        df = query_database('saildata', wind=wind_param, crew=crew_param, vmg_avg=vmg_avg, date=date,
                            current=current, waves=waves, rating=rating, wind_avg=wind_avg)
    else:
        df = query_database('saildata', wind=wind_param, crew=crew_param, vmg_avg=vmg_avg, current=current,
                            waves=waves, rating=rating, wind_avg=wind_avg)
    # Group the data by stream id and session id
    if len(df) < 1:
        st.warning('No data found for the given input')

    groups = df.groupby(['session_id', 'stream_id'])
    df = groups.apply(create_relative_time)
    # Convert the relative time to seconds
    df['rel_time'] = df['rel_time'].dt.total_seconds()
    df['ident'] = df['session_id'].astype(str) + '-' + df['stream_id'].astype(str)
    with plot:
        hover_data_opt = df.columns.tolist()
        hover_data_sel = st.multiselect("Hoverdata", hover_data_opt)
        if datatype == 'Graph':
            if len(df) > 0:


                # create a list of options for the dropdown
                axis_opt = ['vmg', 'Roll', 'TWA', 'Pitch', 'WindSpeed', 'time', 'Yaw', 'SOG', 'COG', 'WindDirection', 'rel_time']

                # create the dropdown using the selectbox function
                yaxis_sel = st.selectbox('Y-axis:', axis_opt, index=axis_opt.index('vmg'))
                xaxis_sel = st.selectbox('X-axis:', axis_opt, index=axis_opt.index('rel_time'))


                # Create the line chart using plotly express
                fig = px.line(df, x=xaxis_sel, y=yaxis_sel, color='ident', hover_name='Crew',
                              hover_data=hover_data_sel)
                # Update the x-axis title
                fig.update_xaxes(title_text=xaxis_sel)
                # Update the y-axis title
                fig.update_yaxes(title_text=yaxis_sel)

                # Show the plot using streamlit
                st.plotly_chart(fig)
            else:
                st.warning('No data found for the given input')

        elif datatype == 'Map':

            # create a list of options for the dropdown
            color_opt = dir(px.colors.sequential)
            color_opt = [item for item in color_opt if '__' not in item and 'swatches' not in item]

            # create the dropdown using the selectbox function
            color_sel = st.selectbox('Color:', color_opt, index=color_opt.index('Turbo'))

            GPSplot = px.scatter_mapbox(df, lat="latitude", lon="longitude", color="vmg",
                                        color_continuous_scale=color_sel,
                                        hover_data=hover_data_sel)
            GPSplot.update_layout(mapbox_style="carto-positron", mapbox_zoom=12,
                                  margin={"r": 0, "t": 0, "l": 0, "b": 0})
            st.plotly_chart(GPSplot)

        elif datatype == 'Polar':
            # create a list of options for the dropdown
            theta_opt = ['TWA', 'Yaw', 'COG', 'WindDirection']
            r_opt = ['vmg', 'Roll', 'Pitch', 'WindSpeed', 'Yaw', 'SOG', 'rel_time']

            # create the dropdown using the selectbox function
            theta_sel = st.selectbox('Theta:', theta_opt, index=theta_opt.index('TWA'))
            r_sel = st.selectbox('Radius:', r_opt, index=r_opt.index('vmg'))

            if theta_sel == 'TWA':
                theta_range = [-180, 180]
            else:
                theta_range = [0, 360]

            PolarPlot = px.scatter_polar(df, r=r_sel, theta=theta_sel, range_theta=theta_range,
                                   start_angle=90, direction='counterclockwise',
                                         hover_data=hover_data_sel)
            st.plotly_chart(PolarPlot)
    with data:
        st.write('Hejssa')
        oversigt_opt = ['By Point', 'By Segment', 'By Session', 'just show me everythang']

        # create the dropdown using the selectbox function
        oversigt_sel = st.radio('Overview:', oversigt_opt, index=oversigt_opt.index('By Segment'), horizontal=True)

        if oversigt_sel == 'By Point':
            inter_columns = ['vmg', 'Roll', 'TWA', 'Pitch', 'WindSpeed', 'time', 'Yaw', 'SOG', 'COG', 'WindDirection']
            st.write(df[inter_columns].describe())
        elif oversigt_sel == 'By Segment':

            grouped_seg_data = df.groupby(['session_id', 'stream_id'])
            #
            # overview_seg_data = grouped_seg_data.Crew

            agg_funcs = {
                'vmg': ['min', 'mean', 'max', 'std'],
                'WindSpeed': ['min', 'mean', 'max', 'std'],
                'Roll': ['min', 'mean', 'max', 'std'],
                'TWA': ['min', 'mean', 'max', 'std'],
                'Pitch': ['min', 'mean', 'max', 'std'],
                'Yaw': ['min', 'mean', 'max', 'std'],
                'SOG': ['min', 'mean', 'max', 'std'],
                'COG': ['min', 'mean', 'max', 'std'],
                'WindDirection': ['min', 'mean', 'max', 'std'],
                'Crew': lambda x: x.iloc[0],
                'Date': lambda x: x.iloc[0],
                'run_length': lambda x: x.iloc[0],
                'Forestay': lambda x: x.iloc[0],
                'Team_weight': lambda x: x.iloc[0],
                'Focus': lambda x: x.iloc[0],
                'Wind': lambda x: x.iloc[0],
                'Waves': lambda x: x.iloc[0],
                'Current': lambda x: x.iloc[0],
                'Rating': lambda x: x.iloc[0],
                'ident': lambda x: x.iloc[0],
            }

            # Apply the aggregation functions to the grouped DataFrame
            grouped_seg_data_agg = grouped_seg_data.agg(agg_funcs)

            # Flatten the column names to make them easier to work with
            grouped_seg_data_agg.columns = ['_'.join(col).strip() for col in grouped_seg_data_agg.columns.values]

            # Reset the index to make the session_id and stream_id columns regular columns
            grouped_seg_data_agg = grouped_seg_data_agg.reset_index()

            # Rename the crew and date columns
            grouped_seg_data_agg = grouped_seg_data_agg.rename(
                columns={'Crew_<lambda>': 'Crew', 'Date_<lambda>': 'Date', 'run_length_<lambda>': 'run_length',
                         'Forestay_<lambda>': 'Forestay', 'Team_weight_<lambda>': 'Team_weight', 'Focus_<lambda>': 'Focus',
                         'Wind_<lambda>': 'Wind', 'Waves_<lambda>': 'Waves', 'Current_<lambda>': 'Current',
                         'Rating_<lambda>': 'Rating', 'ident_<lambda>': 'ident'})

            if "grouped_seg_data_agg_sel" not in st.session_state:
                st.session_state.grouped_seg_data_agg_sel = []

            grouped_seg_data_agg_opt = grouped_seg_data_agg.columns.tolist()

            grouped_seg_data_agg_opt = sorted(grouped_seg_data_agg_opt)

            # grouped_seg_data_agg_default_options = ["vmg_min", "Crew"]

            grouped_seg_data_agg_sel = st.multiselect("Data shown", grouped_seg_data_agg_opt, default=st.session_state.grouped_seg_data_agg_sel)

            st.session_state.grouped_seg_data_agg_sel = grouped_seg_data_agg_sel

            st.write(grouped_seg_data_agg[grouped_seg_data_agg_sel])

        elif oversigt_sel == 'By Session':
            grouped_session_data = df.groupby(['session_id'])

            agg_funcs = {
                'vmg': ['min', 'mean', 'max', 'std'],
                'WindSpeed': ['min', 'mean', 'max', 'std'],
                'Roll': ['min', 'mean', 'max', 'std'],
                'TWA': ['min', 'mean', 'max', 'std'],
                'Pitch': ['min', 'mean', 'max', 'std'],
                'Yaw': ['min', 'mean', 'max', 'std'],
                'SOG': ['min', 'mean', 'max', 'std'],
                'COG': ['min', 'mean', 'max', 'std'],
                'WindDirection': ['min', 'mean', 'max', 'std'],
                'Crew': lambda x: x.iloc[0],
                'Date': lambda x: x.iloc[0],
                'run_length': lambda x: x.iloc[0],
                'Forestay': lambda x: x.iloc[0],
                'Team_weight': lambda x: x.iloc[0],
                'Focus': lambda x: x.iloc[0],
                'Wind': lambda x: x.iloc[0],
                'Waves': lambda x: x.iloc[0],
                'Current': lambda x: x.iloc[0],
                'Rating': lambda x: x.iloc[0],
                'ident': lambda x: x.iloc[0],
            }

            # Apply the aggregation functions to the grouped DataFrame
            grouped_session_data_agg = grouped_session_data.agg(agg_funcs)

            # Flatten the column names to make them easier to work with
            grouped_session_data_agg.columns = ['_'.join(col).strip() for col in grouped_session_data_agg.columns.values]

            # Reset the index to make the session_id and stream_id columns regular columns
            grouped_session_data_agg = grouped_session_data_agg.reset_index()

            # Rename the crew and date columns
            grouped_session_data_agg = grouped_session_data_agg.rename(
                columns={'Crew_<lambda>': 'Crew', 'Date_<lambda>': 'Date', 'run_length_<lambda>': 'run_length',
                         'Forestay_<lambda>': 'Forestay', 'Team_weight_<lambda>': 'Team_weight',
                         'Focus_<lambda>': 'Focus',
                         'Wind_<lambda>': 'Wind', 'Waves_<lambda>': 'Waves', 'Current_<lambda>': 'Current',
                         'Rating_<lambda>': 'Rating', 'ident_<lambda>': 'ident'})

            if "grouped_session_data_agg_sel" not in st.session_state:
                st.session_state.grouped_session_data_agg_sel = []

            grouped_session_data_agg_opt = grouped_session_data_agg.columns.tolist()
            grouped_session_data_agg_opt = sorted(grouped_session_data_agg_opt)

            # grouped_seg_data_agg_default_options = ["vmg_min", "Crew"]

            grouped_session_data_agg_sel = st.multiselect("Data shown", grouped_session_data_agg_opt,
                                                      default=st.session_state.grouped_session_data_agg_sel)

            st.session_state.grouped_session_data_agg_sel = grouped_session_data_agg_sel

            st.write(grouped_session_data_agg[grouped_session_data_agg_sel])




        elif oversigt_sel == 'just show me everythang':
            st.write(df)



    #     if datatype == 'Overview':
    #         crew_map = {'All': None, 'Frederik/Jakob': 'Frederik/Jakob', 'Daniel/Buhl': 'Daniel/Buhl',
    #                     'Kristian/Marcus': 'Kristian/Marcus', 'JC/Mads': 'JC/Mads'}
    #         wind_map = {'All': None, 'Onshore': 'Onshore', 'Offshore': 'Offshore'}
    #         crew_param = crew_map.get(crew)
    #         wind_param = wind_map.get(wind_cond)
    #         if crew_param or wind_param:
    #             available_dates = query_database('unique_dates', crew=crew_param, wind=wind_param)['DATE']
    #         else:
    #             available_dates = query_database('unique_dates')['DATE']
    #         dates = pd.to_datetime(available_dates).dt.strftime("%Y-%m-%d").tolist()
    #         date = st.selectbox('Date', ['All'] + dates) if datatype != 'Date' \
    #             else st.selectbox('Date', ['All'] + dates, index=0)
    #
    #         if date != 'All':
    #             date_sessions = query_database('session_overview', wind=wind_param, crew=crew_param, date=date)
    #             if len(date_sessions) == 0:
    #                 st.warning(f"No sessions on {date} for selected crew/wind condition")
    #             else:
    #                 st.write(date_sessions)
    #         else:
    #             st.write(query_database('session_overview', wind=wind_param, crew=crew_param))
    #
    #     elif datatype == 'All':
    #         st.write(query_database('saildata'))
    #     elif datatype == 'Runs':
    #         st.write(query_database('run_overview'))
elif options == 'Home':
    st.write('Velkommen')

    # Load an image from a file path
    image = 'example.png'

    # Display the image in the Streamlit app
    st.image(image, caption='Example image', use_column_width=True)

if options == 'Upload':
    st.header("Upload Files")

    uploaded_csv_file_1 = st.file_uploader("Upload Sailmon CSV file", type=["csv"], key=st.session_state.state["widget_key"] + "_csv1")
    uploaded_csv_file_2 = st.file_uploader("Upload WindBot CSV file", type=["csv"], key=st.session_state.state["widget_key"] + "_csv2")

    if uploaded_csv_file_1 and uploaded_csv_file_2:
        st.success("All files have been uploaded successfully!")

        # Read and display the content of the first CSV file
        df_csv_1 = pd.read_csv(uploaded_csv_file_1)
        st.write("Content of the first CSV file:")
        st.write(df_csv_1)

        # Read and display the content of the second CSV file
        df_csv_2 = pd.read_csv(uploaded_csv_file_2)
        st.write("Content of the second CSV file:")
        st.write(df_csv_2)

        # load the trained model from a file
        SegPredModel = joblib.load('BIGMODE.joblib')

        SM = SM_handling.upload_session_step1(SM_Rawfile=df_csv_1, Wind_Rawfile=df_csv_2, _SegmentPredictionmodel=SegPredModel)

        vmg_min = st.number_input('Minimum VMG Average:', value=4.0, step=0.1)
        length_min = st.number_input('Minimum Segment Length', value=20, step=5)

        SM = SM_handling.join_rolling_lines(SM, vmg=vmg_min, length=length_min)

        # create a list of options for the dropdown
        color_opt = dir(px.colors.sequential)
        color_opt = [item for item in color_opt if '__' not in item and 'swatches' not in item]

        # create the dropdown using the selectbox function
        color_sel = st.selectbox('Color:', color_opt, index=color_opt.index('Turbo'))

        GPSplot = px.scatter_mapbox(SM, lat="latitude", lon="longitude", color="vmg",
                                    color_continuous_scale=color_sel)
        GPSplot.update_layout(mapbox_style="carto-positron", mapbox_zoom=12,
                              margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(GPSplot)

        st.header("Create Log")
        old_new_crew_opt = ['Old', 'New']
        old_new_crew_sel = st.radio('Crew:', options=old_new_crew_opt, horizontal=True,
                                    index=old_new_crew_opt.index('Old'))

        if old_new_crew_sel == 'New':
            Crew = st.text_input("Type crew:")
        elif old_new_crew_sel == 'Old':
            Crew = st.radio('Select crew', options=SM_handling.fetch_crew(),
                            horizontal=True)

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            current_opt = ['0 - None', '1', '2 - Worst']
            current_sel = st.radio('Current', options=current_opt)

        with col2:
            wave_opt = ['0 - None', '1', '2', '3 - Worst']
            wave_sel = st.radio('Waves', options=wave_opt)

        with col3:
            rating_opt = ['1 - Worst', '2', '3', '4', '5 - Best']
            rating_sel = st.radio('Rating', options=rating_opt)

        with col4:
            Focus_opt = ['Race', 'Other']
            Focus_sel = st.radio('Focus:', options=Focus_opt, index=Focus_opt.index('Other'))

        with col5:
            Wind_opt = ['Onshore', 'Offshore']
            Wind_sel = st.radio('Focus:', options=Wind_opt, index=Wind_opt.index('Onshore'))

        with col6:
            Forestay = st.number_input('Forestay length:', value=430, step=5)
            Team_weight = st.number_input('Team weight:', value=165, step=1)

        col7, col8 = st.columns(2)

        with col7:
            Wind_range_min = st.number_input('Minimum wind range:', value=5, step=1)

        with col8:
            Wind_range_max = st.number_input('Maximum wind range:', value=15, step=1)

        notes = st.text_input("Notes:")

        st.write('Log Preview:')

        if type(current_sel) != int:
            match = re.search(r'\d+', current_sel)
            current_sel = int(match.group())

        if type(wave_sel) != int:
            match = re.search(r'\d+', wave_sel)
            wave_sel = int(match.group())

        if type(rating_sel) != int:
            match = re.search(r'\d+', rating_sel)
            rating_sel = int(match.group())

        session_overview = pd.DataFrame([Crew, Forestay, Team_weight, Focus_sel, Wind_sel, Wind_range_min,
                                         Wind_range_max, wave_sel, current_sel, rating_sel, notes],
                                        ['Crew', 'Forestay', 'Team_weight', 'Focus', 'Wind', 'Wind_range_min',
                                         'Wind_range_max', 'Waves', 'Current', 'Rating', 'Notes'])

        session_overview = session_overview.transpose()

        delta = SM.time.iloc[-1] - SM.time.iloc[0]

        session_overview['Date'] = SM.time.iloc[0].date()
        session_overview['session_length'] = delta.total_seconds()
        session_overview['Median_Wind'] = SM.WindSpeed.median()

        st.write(session_overview)

        # establish a connection to the MySQL server for mysql.connecter
        # connect to the database
        # cnx = mysql.connector.connect(user='root', password='49er',
        #                               host='127.0.0.1', database='bac')
        cnx = mysql.connector.connect(user='root', password='49erproject',
                                      host='127.0.0.1', port=3306, database='bac')

        # create a cursor object
        cursor = cnx.cursor()

        # execute the query
        query = ("SELECT MAX(session_id) FROM saildata")
        cursor.execute(query)
        # fetch the results
        results = cursor.fetchall()

        Session_id = results[0][0] + 1

        session_overview['session_id'] = Session_id

        # close the cursor and connection
        cursor.close()
        cnx.close()

        st.write(session_overview)


if options == 'Draft':
    def my_function():
        st.write("Button clicked!")


    # create the button
    if st.button("Click me!"):
        my_function()


st.stop()
