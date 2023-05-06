import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt


# Function to save uploaded file to directory
def save_uploaded_file(uploadedfile, file_ex, file_id):
    folder_name = 'Sailmon' if uploadedfile.name.startswith('SM') else 'Log'
    if not os.path.exists('Bibliotek/{}'.format(folder_name)):
        os.makedirs('Bibliotek/{}'.format(folder_name))

    if file_ex == 'xlsx':
        read_file = pd.read_excel(uploadedfile)
        new_name = file_id + '.csv'
        read_file.to_csv('Bibliotek/{}/{}'.format(folder_name, new_name), index=False)
        df = pd.read_csv('Bibliotek/{}/{}'.format(folder_name, new_name), delimiter=',')
        column_names = df.columns.tolist()
        df = df.rename(columns=dict(zip(column_names, [c.strip('"') for c in column_names])))
        return st.sidebar.success('Saved file: {} in Bibliotek/{}'.format(new_name, folder_name))

    elif file_ex == 'csv':
        with open(os.path.join('Bibliotek', folder_name, uploadedfile.name), 'wb') as f:
            f.write(uploadedfile.getbuffer())
        df = pd.read_csv(os.path.join('Bibliotek', folder_name, uploadedfile.name))
        column_names = df.columns.tolist()
        df = df.rename(columns=dict(zip(column_names, [c.strip('"') for c in column_names])))
        return st.sidebar.success('Saved file: {} in Bibliotek/{}'.format(uploadedfile.name, folder_name))


def save_uploaded_image(uploadedfile):
    folder_name = 'Images_wind'
    if not os.path.exists('Bibliotek/{}'.format(folder_name)):
        os.makedirs('Bibliotek/{}'.format(folder_name))
    plt.savefig('Bibliotek/{}/{}'.format(folder_name, uploadedfile.name))
    return st.sidebar.success('Saved file: {} in Bibliotek/{}'.format(uploadedfile.name, folder_name))
