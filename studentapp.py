import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv(r"student_info.csv")

choice=st.sidebar.radio("Linear regression model",["DataFrame","Graph","About","Result"],index=0)
 
if choice=="About":
    st.title("About")
    st.header("Columns present in dataset")
    st.write(df.columns)
    st.subheader("first 10 data")
    st.write(df.head(10))
    st.subheader("descibe")
    st.write(df.describe())



df2=df.fillna(df.mean())

x=df2.drop("student_marks",axis="columns")
y=df2.drop("study_hours",axis="columns")



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=51)



le=LinearRegression()
le.fit(x_train,y_train)

y_pred=le.predict(x_test)

if choice=="Graph":
    st.title("Grpahs")
    st.header("Original graph")
    plt.figure(figsize=(8,8))
    plt.scatter(x=df.study_hours,y=df.student_marks)
    plt.xlabel("study_hours")
    plt.ylabel("student_marks")
    plt.title("scatterplot student_hourse vs student_marks")
    st.pyplot()

    st.header("Trained graph")
    plt.scatter(x_train,y_train)
    st.pyplot()

    st.header("Model graph")
    plt.scatter(x_test,y_test)
    plt.plot(x_train,le.predict(x_train),color='r')
    st.pyplot()

if choice=="DataFrame":
    st.title("DataFrame")
    st.header("predicted data frame")
    st.write(pd.DataFrame(np.c_[x_test,y_test,y_pred],columns = ["student_study_hours","student_marks_original","predicted_student_marks"]))
    st.write("accuracy of the predicted data :",le.score(x_test,y_test))



if choice=="Result":
    st.title("Result")
    import joblib
    joblib.dump(le,"student_mark_prediction_model.pkl")
    model=joblib.load("student_mark_prediction_model.pkl")
    st.title("get result")
    n=st.number_input("enter how many hours has to study")
    if st.button("get result"):
        st.write("predicted result is :",model.predict([[n]])[0][0])
