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
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import datetime
from random import randint
from file_handling import *
import openpyxl
import joblib
import re


# establish a connection to the MySQL server for mysql.connecter
# connect to the database
cnx = mysql.connector.connect(user='root', password='49erproject',
                              host='35.228.102.18', database='bac')
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

print(list_of_crew)
