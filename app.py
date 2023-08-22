import streamlit as st
import pandas as pd
import os
#profiling
import pandas_profiling    #import pandas profiling
from streamlit_pandas_profiling import st_profile_report
#ml packages

from pycaret.regression import setup,  compare_models, pull, save_model

##adding a sidebar
#automated profiling using Pandas profiling

with st.sidebar:
    st.image("https://littleml.files.wordpress.com/2019/09/automl-2.png?w=584&h=269")
    st.title("AutoML Application")
    choice = st.radio("Navigation", ["Upload Data", "Auto Profiling", "ML", "Download"])
    st.info("Build your own automated Machine Learning pipeline using Streamlit, Pandas Profiling and PyCaret.")
st.write("Hello World")

if os.path.exists("source_data.csv"):
    df = pd.read_csv("source_data.csv", index_col = None)


if choice == "Upload Data":
    st.title("Upload your Data")
    file = st.file_uploader("Upload your Dataset here")
    if file:
        df = pd.read_csv(file, index_col = None)
        df.to_csv('source_data.csv', index = None)  #writing source data to local machine from application
        st.dataframe(df)

if choice == "Auto Profiling":
    st.title("Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == "ML":
    st.title("Automated Machine Learning!")
    target = st.selectbox("Select target Variable", df.columns)
    if st.button("Train Model"):
        setup(df, target = target)
        setup_df = pull()
        st.info("ML experiment setting")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("Best Model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'best_model')

if choice == "Download":
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download the best model", f, "best_model.pkl")