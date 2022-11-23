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
    
    st.write("##KNN")
    K=10
    knn=KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    
    skor_akurasi = round(100 * accuracy_score(y_test,y_pred))
    st.write("Model accuracy score : {0:0.2f}" . format(skor_akurasi))

    st.write("##Decision Tree")
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    # prediction
    dt.score(X_test, y_test)
    y_pred = dt.predict(X_test)
    
    #Accuracy
    akurasi = round(100 * accuracy_score(y_test,y_pred))
    st.write('Model Accuracy Score: {0:0.2f}'.format(akurasi))
    
with tab5:
    <form>
      <fieldset><legend>Tambah Data Nasabah</legend>
        <table>
          <tr>
            <th><label for="field1"><span>Umur <span class="required">*</span></span></th>
            <td><input type="number" class="input-field" name="field1" value="age" /></label></td>
          </tr>
          <tr>
            <th><label for="field2"><span>Jenis Kelamin<span class="required">*</span></span></th>
            <td><select name="field4" class="select-field">
            <option value="Female">Perempuan</option>
            <option value="Male">Laki-laki</option>
            </select></label></td>
          </tr>
          <tr>
            <th><label for="field3"><span>on thyroxine<span class="required">*</span></span></th>
            <td><input type="text" class="input-field" name="field3" value="on thyroxine" /></label></td>
          </tr>
          <tr>
            <th><label for="field3"><span>query on thyroxine<span class="required">*</span></span></th>
            <td><input type="text" class="input-field" name="field3" value="query on thyroxine" /></label></td>
          </tr>
		  <tr>
            <th><label for="field3"><span>on antithyroid medication<span class="required">*</span></span></th>
            <td><input type="text" class="input-field" name="field3" value="on antithyroid medication" /></label></td>
          </tr>
		  <tr>
            <th><label for="field3"><span>query on thyroxine<span class="required">*</span></span></th>
            <td><input type="text" class="input-field" name="field3" value="query on thyroxine" /></label></td>
          </tr>
		  <tr>
            <th><label for="field3"><span>query on thyroxine<span class="required">*</span></span></th>
            <td><input type="text" class="input-field" name="field3" value="query on thyroxine" /></label></td>
          </tr>
		  <tr>
            <th><label for="field3"><span>query on thyroxine<span class="required">*</span></span></th>
            <td><input type="text" class="input-field" name="field3" value="query on thyroxine" /></label></td>
          </tr>
		  <tr>
            <th><label for="field3"><span>query on thyroxine<span class="required">*</span></span></th>
            <td><input type="text" class="input-field" name="field3" value="query on thyroxine" /></label></td>
          </tr>
          <tr>
            <th><label for="field4"><span>Rata-Rata Overdue<span class="required">*</span></span>
            <td><select name="field4" class="select-field"></th>
            <option value="1">0 - 30 hari</option>
            <option value="2">31 - 45 hari</option>
            <option value="3">46 - 60 hari</option>
            <option value="4">61 - 90 hari</option>
            <option value="5"> Lebih dari 90 hari</option>
            </select></label>
            </td>
          </tr>
      </table>
      <label><span>&nbsp;</span><input type="submit" value="Submit"/></label>
      </fieldset>
      </form>
    
    
    
    
