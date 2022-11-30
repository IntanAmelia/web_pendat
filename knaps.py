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
# Penambangan Data
""")

st.write("=========================================================================")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Description", "Import Data", "Preprocessing", "Modelling", "Implementation"])

with tab1:
    st.write("Dataset yang digunakan pada penelitian ini yakni Thyroid Disease Datasets")
    st.write("Thyroid Disease")
    st.write("The most common thyroid disorder is hypothyroidism. Hypo- means deficient or under(active), so hypothyroidism is a condition in which the thyroid gland is underperforming or producing too little thyroid hormone. Recognizing the symptoms of hypothyroidism is extremely important.")
    st.write("Data Set Information : From Garavan Institute, Documentation: as given by Ross Quinlan, 6 databases from the Garavan Institute in Sydney, Australia.")
    st.write("Approximately the following for each database : 2800 training (data) instances and 972 test instances, Plenty of missing data, 29 or so attributes, either Boolean or continuously-valued")
    st.write("https://www.kaggle.com/datasets/yasserhessein/thyroid-disease-data-set")
    
with tab2:
    st.write("Load Data :")
    data_file = st.file_uploader("Upload CSV",type=['csv'])
	if st.button("Process"):
		if data_file is not None:
			file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
			st.write(file_details)
			df = pd.read_csv(data_file)
			st.dataframe(df)
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
       
    y=data['binaryClass']
    x=data.drop(['binaryClass'],axis=1)
    st.write("Menampilkan data yang sudah dinormalisasi dan dilakukan scaled features")
    st.dataframe(data)
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test= train_test_split(x,y,test_size=0.3,stratify=y)
    st.write("X_train.shape")
    st.write(X_train.shape)
    st.write("X_test.shape")
    st.write(X_test.shape)
    st.write("y_train.shape")
    st.write(y_train.shape)
    st.write("y_test.shape")
    st.write(y_test.shape)

with tab4:
    st.write("## Naive Bayes")
    # Feature Scaling to bring the variable in a single scale
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    GaussianNB(priors=None)
    
    # Fitting Naive Bayes Classification to the Training set with linear kernel
    nvklasifikasi = GaussianNB()
    nvklasifikasi = nvklasifikasi.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = nvklasifikasi.predict(X_test)
    y_pred
    
    #lets see the actual and predicted value side by side
    y_compare = np.vstack((y_test,y_pred)).T
    #actual value on the left side and predicted value on the right hand side
    #printing the top 5 values
    y_compare[:5,:]
    
    # Menentukan probabilitas hasil prediksi
    nvklasifikasi.predict_proba(X_test)
    akurasi = round(100 * accuracy_score(y_test, y_pred))
    st.write('Model accuracy score: {0:0.2f}'. format(akurasi))
    
    # print the scores on training and test set
    akurasi_training = round(100* nvklasifikasi.score(X_train, y_train))
    akurasi_test = round(100 * nvklasifikasi.score(X_test, y_test) )
    st.write('Training set score: {:.2f}'.format(akurasi_training))
    st.write('Test set score: {:.2f}'.format(akurasi_test))
    
    st.write("## KNN")
    K=10
    knn=KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    
    skor_akurasi = round(100 * accuracy_score(y_test,y_pred))
    st.write("Model accuracy score : {0:0.2f}" . format(skor_akurasi))

    st.write("## Decision Tree")
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    # prediction
    dt.score(X_test, y_test)
    y_pred = dt.predict(X_test)
    
    #Accuracy
    akurasi = round(100 * accuracy_score(y_test,y_pred))
    st.write('Model Accuracy Score: {0:0.2f}'.format(akurasi))
    
with tab5:
    Age = st.number_input('Masukkan Umur Anda : ', 0,1000)
    Gender = st.radio('Pilih Gender',['Male','Female'])
    On_thyroxine = st.radio('On thyroxine',['f','t'])
    Query_on_thyroxine = st.radio('Query on thyroxine',['f','t'])
    On_antithyroid_medication = st.radio('On antithyroid medication',['f','t'])
    sick = st.radio('Sick',['f','t'])
    pregnant = st.radio('Pregnant',['f','t'])
    Thyroid_surgery = st.radio('Thyroid surgery',['f','t'])
    I131_treatment = st.radio('I131 treatment',['f','t'])
    Query_hypothyroid = st.radio('Query hypothyroid',['f','t'])
    Query_hyperthyroid = st.radio('Query hyperthyroid',['f','t'])
    Lithium = st.radio('Lithium',['f','t'])
    Goitre = st.radio('Goitre',['f','t'])
    Tumor = st.radio('Tumor',['f','t'])
    Hypopituitary = st.radio('Hypopituitary',['f','t'])
    Psych = st.radio('Psych',['f','t'])
    TSH_Measured = st.radio('TSH measured',['f','t'])
    TSH = st.number_input('TSH : ', -1,1000)
    T3_Measured = st.radio('T3 measured',['f','t'])    
    T3 = st.number_input('T3 : ', -1,1000)
    TT4_Measured = st.radio('TT4 measured',['f','t'])
    TT4 = st.number_input('TT4 : ', -1,1000)
    T4U_Measured = st.radio('T4U measured',['f','t'])
    T4U = st.number_input('T4U : ', -1,1000)
    FTI_Measured = st.radio('FTI measured',['f','t'])
    FTI = st.number_input('FTI : ', -1,1000)
    TBG_Measured = st.radio('TBG measured',['f','t'])
    TBG = st.text_input('TBG')
    Referral_source = st.multiselect('Referral source',['SVHC', 'SVI', 'STMW', 'SVHD', 'Other'])
    
    
    features = {'age' : Age, 'sex' : Gender, 'on thyroxine' : On_thyroxine, 'query on thyroxine' : Query_on_thyroxine, 'on antithyroid medication' : On_antithyroid_medication, 'sick' : sick, 'pregnant' : pregnant, 'thyroid surgery' : Thyroid_surgery , 'I131 treatment' : I131_treatment, 'query hypothyroid' : Query_hypothyroid, 'query hyperthyroid' : Query_hyperthyroid, 'lithium' : Lithium, 'goitre' : Goitre, 'tumor' : Tumor, 'hypopituitary' : Hypopituitary, 'psych' : Psych, 'TSH Measured' : TSH_Measured, 'TSH' : TSH, 'T3 Measured' : T3_Measured, 'T3' : T3, 'TT4 Measured' : TT4_Measured, 'TT4' : TT4, 'T4U Measured' : T4U_Measured, 'T4U' : T4U, 'FTI Measured' : FTI_Measured, 'FTI' : FTI, 'TBG Measured' : TBG_Measured, 'TBG' : TBG, 'referral source' : Referral_source}        

    features_df  = pd.DataFrame([features])

    st.table(features_df) 

    if st.button('Prediksi'):
        features_dfd = np.reshape(features_df, (1, -1))
        st.dataframe(features_dfd)
        
        from sklearn.preprocessing import LabelEncoder
        enc=LabelEncoder()
        for x in features_dfd.columns:
          features_dfd[x]=enc.fit_transform(features_dfd[x])
        features_dfd.info()

        features_dfd.head()
        st.dataframe(features_dfd)

        features_dfd['age']=(features_dfd['age']-features_dfd['age'].min())/(features_dfd['age'].max()-features_dfd['age'].min())
        features_dfd['TT4']=(features_dfd['TT4']-features_dfd['TT4'].min())/(features_dfd['TT4'].max()-features_dfd['TT4'].min())
        features_dfd['T4U']=(features_dfd['T4U']-features_dfd['T4U'].min())/(features_dfd['T4U'].max()-features_dfd['T4U'].min())
        features_dfd['FTI']=(features_dfd['FTI']-features_dfd['FTI'].min())/(features_dfd['FTI'].max()-features_dfd['FTI'].min())


        st.dataframe(features_dfd)

        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test= train_test_split(x,y,test_size=0.3,stratify=y)
        
        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)
        # prediction
        dt.score(X_test, y_test)
        y_pred = dt.predict(X_test)

        #Accuracy
        akurasi = round(100 * accuracy_score(y_test,y_pred))
        
        # Custom value to predict
        result_test_decision_tree = decision_tree.predict(features_dfd)
        print(f"Customer : Memiliki Hasil Binary Class {result_test_decision_tree[no_index]} Pada metode Decision Tree model")
        
