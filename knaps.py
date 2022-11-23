import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC

st.write(""" 
# Cek data
""")

st.write("=========================================================================")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Description", "Import Data", "Preprocessing", "Modelling", "Evalutions"])

with tab1:
    st.write("Dataset yang digunakan pada penelitian ini yakni Thyroid Disease Datasets")
    st.write("Thyroid Disease")
    st.write("The most common thyroid disorder is hypothyroidism. Hypo- means deficient or under(active), so hypothyroidism is a condition in which the thyroid gland is underperforming or producing too little thyroid hormone. Recognizing the symptoms of hypothyroidism is extremely important.")
    st.write("Data Set Information : From Garavan Institute, Documentation: as given by Ross Quinlan, 6 databases from the Garavan Institute in Sydney, Australia.")
    st.write("Approximately the following for each database : 2800 training (data) instances and 972 test instances, Plenty of missing data, 29 or so attributes, either Boolean or continuously-valued")
    st.write("https://www.kaggle.com/datasets/yasserhessein/thyroid-disease-data-set")
    
with tab2:
    st.write("Load Data")
    data = pd.read_csv("https://raw.githubusercontent.com/IntanAmelia/web_pendat/main/hypothyroid.csv")
    st.dataframe(data)

with tab3:
    st.write("Normalisasi Data")
    data.head()
    
    from sklearn.preprocessing import LabelEncoder
    enc=LabelEncoder()
    for x in data.columns:
      data[x]=enc.fit_transform(data[x])
    data.info()
    
    data.head()
    st.dataframe(data)
    
    st.write("Scaled Features")
    data['age']=(data['age']-data['age'].min())/(data['age'].max()-data['age'].min())
    data['TT4']=(data['TT4']-data['TT4'].min())/(data['TT4'].max()-data['TT4'].min())
    data['T4U']=(data['T4U']-data['T4U'].min())/(data['T4U'].max()-data['T4U'].min())
    data['FTI']=(data['FTI']-data['FTI'].min())/(data['FTI'].max()-data['FTI'].min())
    
    st.dataframe(data)
    
    data = data.drop(columns = "binaryClass")

    st.write("Menampilkan data yang sudah dinormalisasi dan dilakukan scaled features")
    st.dataframe(data)
    
    percent_amount_of_test_data = 0.3

    st.write("## Hitung Data")
    st.write("- Ambil kolom 'Binary Class' sebagai target kolom untuk kategori kelas")
    st.write("- Pisahkan data latih dengan data tes")
    st.write("""            Spliting Data

                data latih (nilai data)
                X_train 

                data tes (nilai data)
                X_test 

                data latih (kelas data)
                y_train

                data tes (kelas data)
                y_test""")

    # separate target 

    # values
    matrices_X = data.iloc[:,0:29].values

    # classes
    matrices_Y = data.iloc[:,30].values

    X_1 = data.iloc[:,0:29].values
    Y_1 = data.iloc[:, 30].values

    # X_train, X_test, y_train, y_test = train_test_split(matrices_X, matrices_Y, test_size = percent_amount_of_test_data, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X_1, Y_1, test_size = percent_amount_of_test_data, random_state=0)

    st.write("Menampilkan Y_1")
    st.write(Y_1)
    
    st.write("Menampilkan X_1")
    st.write(X_1)
    
    
