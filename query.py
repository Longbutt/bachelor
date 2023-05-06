import mysql.connector
import pandas as pd
import streamlit as st


@st.cache
def query_database(query_type, wind=None, crew=None, date=None, vmg_avg=None, waves=None, current=None, rating=None,
                   wind_min=None, wind_max=None, wind_avg=None):
    # establish a connection to the MySQL server for mysql.connecter
    # cnx = mysql.connector.connect(user='root', password='49er', host='127.0.0.1', database='bac')

    cnx = mysql.connector.connect(user='root', password='49erproject',
                                  host='127.0.0.1', port=3306, database='bac')

    # create a cursor object
    cursor = cnx.cursor()

    if query_type == 'session_overview':
        # Query 1: Get all data from session_overview table
        query = """
        SELECT *
        FROM session_overview
        """
        if wind:
            query += f"WHERE Wind = '{wind}'"
        if crew:
            if wind:
                query += f" AND Crew = '{crew}'"
            else:
                query += f"WHERE Crew = '{crew}'"
        if date:
            if wind or crew:
                query += f" AND DATE = '{date}'"
            else:
                query += f"WHERE DATE = '{date}'"

    elif query_type == 'unique_dates':
        query = "SELECT DISTINCT DATE FROM session_overview"

    elif query_type == 'run_overview':
        # Query 2: Get all data from run_overview table
        query = """
        SELECT *
        FROM run_overview
        """
        if wind:
            query += f"WHERE Median_Wind >= {wind}"
        if crew:
            if wind:
                query += f" AND Crew = '{crew}'"
            else:
                query += f"WHERE Crew = '{crew}'"
        if date:
            if wind or crew:
                query += f" AND DATE(Session_Start) = '{date}'"
            else:
                query += f"WHERE DATE(Session_Start) = '{date}'"

    elif query_type == 'saildata':
        # Query 3: Get all data possible
        query = """
        SELECT sd.time, sd.Roll, sd.Pitch, sd.WindSpeed, sd.WindDirection, ro.avg_VMG, sd.vmg, sd.TWA, so.Crew, 
        so.Forestay, so.Team_weight, so.Focus, so.Wind, so.Waves, so.Current, so.Rating, sd.stream_id, sd.session_id, 
        sd.latitude, sd.longitude, sd.Yaw, sd.COG, sd.SOG, so.Date, so.session_length, ro.run_length
        FROM saildata sd
        JOIN run_overview ro ON sd.stream_id = ro.stream_id
        JOIN session_overview so ON sd.session_id = so.session_id
        """
        if wind:
            query += f"WHERE so.Wind = '{wind}'"
        if crew:
            if wind:
                query += f" AND so.Crew = '{crew}'"
            else:
                query += f"WHERE so.Crew = '{crew}'"
        if date:
            if wind or crew:
                query += f" AND so.DATE = '{date}'"
            else:
                query += f"WHERE so.DATE = '{date}'"
        if waves:
            if wind or crew or date:
                query += f" AND so.Waves = {waves}"
            else:
                query += f" WHERE so.Waves = {waves}"
        if current:
            if wind or crew or date or waves:
                query += f" AND so.Current = {current}"
            else:
                query += f" WHERE so.Current = {current}"
        if rating:
            if wind or crew or date or waves or current:
                query += f" AND so.Rating = {rating}"
            else:
                query += f" WHERE so.Rating = {rating}"
        if vmg_avg:
            if wind or crew or date or waves or current or rating:
                query += f" AND ro.avg_VMG > {vmg_avg}"
            else:
                query += f" WHERE ro.avg_VMG > {vmg_avg}"
        if wind_min:
            if wind or crew or date or waves or current or rating or vmg_avg:
                query += f" AND so.Wind_range_min > {wind_min}"
            else:
                query += f" WHERE so.Wind_range_min > {wind_min}"
        if wind_max:
            if wind or crew or date or waves or current or rating or vmg_avg or wind_min:
                query += f" AND so.Wind_range_max < {wind_max}"
            else:
                query += f" WHERE so.Wind_range_max < {wind_max}"
        if wind_avg:
            if wind or crew or date or waves or current or rating or vmg_avg or wind_min or wind_max:
                query += f" AND ro.avg_WindSpeed > {wind_avg}"
            else:
                query += f" WHERE ro.avg_WindSpeed > {wind_avg}"

        query += " AND CONCAT(sd.stream_id, sd.session_id) = CONCAT(ro.stream_id, ro.session_id)"

    else:
        raise ValueError("Invalid query type. Choose from 'session_overview', 'run_overview', or 'saildata'.")

    # execute the query
    cursor.execute(query)
    column_names = [i[0] for i in cursor.description]
    # fetch the results
    results = cursor.fetchall()
    # create a dataframe
    df = pd.DataFrame(results, columns=column_names)

    # close the cursor and connection
    cursor.close()
    cnx.close()

    return df
