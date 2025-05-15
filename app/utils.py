import pandas as pd
import streamlit as st

@st.cache_data
def load_data(csv_file):
    """
    Load data from a CSV file into a pandas DataFrame.
    """
    return pd.read_csv(csv_file)