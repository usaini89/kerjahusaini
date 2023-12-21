import streamlit as st 
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

st.write("# advertising")
st.write("This app predicts the **SALES** type!")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('tv', 4.3, 7.9, 5.4) #MASUKAN YG MINIMUM AMOUNT DULU DAN YG MAX DAN YG DEFOLT
    #BOLEH UBAH SEPAL LENG SEBGAI NME LAIN
    sepal_width = st.sidebar.slider('radio', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('N.P', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length, # SUSUNAN MESTI IKUT YG ATAS 
            'sepal_width': sepal_width, # SUSUNAN MESTI IKUT YG ATAS 
            'petal_length': petal_length, # SUSUNAN MESTI IKUT YG ATAS 
            'petal_width': petal_width} # SUSUNAN MESTI IKUT YG ATAS 
    features = pd.DataFrame(data, index=[0]) 
    # 1 ROW DATA INDEX 0
    return features
   # FINAL DIA AKAN RETURN
df = user_input_features()    # CALL FUNCTION

st.subheader('User Input parameters') # UTK TULIS PARAMWETER
st.write(df)

data = sns.load_dataset('iris') #LOAD DATA IRIS
X = data.drop(['species'],axis=1)
Y = data.species.copy()

#NO FEUTERE SCALING ANYMORE


modelGaussianIris = GaussianNB()
modelGaussianIris.fit(X, Y) #TRAINING

prediction = modelGaussianIris.predict(df) #USER INPUT
prediction_proba = modelGaussianIris.predict_proba(df) # PREDICT PROBILITY

st.subheader('Class labels and their corresponding index number')
st.write(Y.unique())

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
