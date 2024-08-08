import pickle
import streamlit as st
import numpy as np 
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Loading the saved models

diabetes_model = pickle.load(open("Saved_Models\\diabetes_model.sav",'rb'))

heart_model = pickle.load(open("Saved_Models\\heart_model.sav",'rb'))

parkinson_model = pickle.load(open("Saved_Models\\Parkinson_model.sav",'rb'))

# Define functions for each page
def heart():
    st.title('Heart Disease Prediction Using ML')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age = st.text_input('Age of the Person')

    with col2:
        sex = st.selectbox("Select your Gender", ["Male","Female"])

        if(sex == "Male"):
            sex = 1
        else:
            sex = 0

    with col3:
        cp = st.selectbox("Chest Pain Level",[0,1,2,3])

    with col4: 
        trestbps = st.text_input('resting blood pressure')

    with col1:
        chol = st.text_input('serum cholestoral in mg/dl')

    with col2:
        fbs = st.selectbox('fasting blood sugar',["yes","no"])

        if(fbs == "yes"):
            fbs = 1
        else:
            fbs = 0

    with col3:
        restecg = st.selectbox('resting electrocardiographic results',[0,1,2])
    
    with col4:
        thalach = st.text_input('maximum heart rate achieved')

    with col1:
        exang = st.selectbox('Exercise Induced Angina',["yes","no"])

        if(exang == "yes"):
            exang = 1
        else:
            exang = 0

    with col2:
        oldpeak = st.text_input('ST depression induced by exercise relative to rest')
    
    with col3:
        slope = st.selectbox("The slope of the peak exercise ST segment",[0,1,2])

    with col4:
        ca = st.selectbox("Number of major vessels (0-3) colored by Flourosopy",[0,1,2,3])
    
    with col1:
        thal = st.selectbox("Thalium stress result",[i for i in range(1,8)])

    # code for prediction
    heart_diagnosis = ""

    # creating a button for prediction

    if st.button('Heart Test Result'):
        heart_data = [age,sex,cp,trestbps,chol,fbs,restecg,thalach, exang,oldpeak,slope,ca,thal]
        
        heart_data = [float(x) for x in heart_data]

        heart_prediction = heart_model.predict([heart_data])

        if(heart_prediction[0] == 1):
            heart_diagnosis = "The Person has Heart Disease"
        else:
            heart_diagnosis = "The Person doesn't have heart Disease"

        st.success(heart_diagnosis)

def diabetes():
    st.title('Diabetes Prediction Using ML')

    col1, col2, col3 = st.columns(3)

    diab_df = pd.read_csv("dataset\\diabetes.csv")

    diab_X = diab_df.drop('Outcome',axis=1)

    diab_scaler = StandardScaler()
    diab_scaler = diab_scaler.fit(diab_X)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure Value')

    with col1: 
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')

    
    # code for prediction
    diab_diagnosis = ''

    # creating a button for prediction

    if st.button('Diabetes Test Result'):
        diab_data = [[Pregnancies, Glucose,BloodPressure,SkinThickness,Insulin, BMI, DiabetesPedigreeFunction, Age]]
        diab_data = np.array(diab_data)
        diab_data = diab_scaler.transform(diab_data)

        diab_prediction = diabetes_model.predict(diab_data)

        if(diab_prediction[0] == 1):
            diab_diagnosis = 'The Person is Diabetic'
        else:
            diab_diagnosis = 'The Person is not Diabetic'

        st.success(diab_diagnosis)

def Parkinson():
    st.title("Parkinson's Disease Prediction using ML")

    park_df = pd.read_csv("dataset\\parkinsons.data")
    park_X = park_df.drop(columns=['name','status'],axis=1)

    park_scaler = StandardScaler()
    park_scaler = park_scaler.fit(park_X)

    col1, col2, col3 = st.columns(3)

    with col1:
        fo = st.text_input("MDVP : Fo(Hz)")

    with col2:
        fhi = st.text_input('MDVP : Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP : Flo(Hz)')

    with col1:
        Jitter_percent = st.text_input('MDVP : Jitter(%)')

    with col2:
        Jitter_Abs = st.text_input('MDVP : Jitter(Abs)')

    with col3:
        RAP = st.text_input('MDVP : RAP')

    with col1:
        PPQ = st.text_input('MDVP : PPQ')

    with col2:
        DDP = st.text_input('Jitter : DDP')

    with col3:
        Shimmer = st.text_input('MDVP : Shimmer')

    with col1:
        Shimmer_dB = st.text_input('MDVP : Shimmer(dB)')

    with col2:
        APQ3 = st.text_input('Shimmer : APQ3')

    with col3:
        APQ5 = st.text_input('Shimmer : APQ5')

    with col1:
        APQ = st.text_input('MDVP : APQ')

    with col2:
        DDA = st.text_input('Shimmer : DDA')

    with col3:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col1:
        spread1 = st.text_input('spread1')

    with col2:
        spread2 = st.text_input('spread2')

    with col3:
        D2 = st.text_input('D2')

    with col1:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        user_input = park_scaler.transform([user_input])


        parkinsons_prediction = parkinson_model.predict(user_input)

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)

# Create a dictionary to map page names to functions
pages = {
    "Heart Disease Prediction": heart,
    "Diabetes Prediction": diabetes,
    "Parkinson Prediction": Parkinson
}

# Add a radio button to the sidebar for navigation
st.sidebar.markdown("# Multiple Disease Prediction System")
selected_page = st.sidebar.radio("Select Which Disease you want to Predict.", options=pages.keys())

# Call the function corresponding to the selected page
pages[selected_page]()
