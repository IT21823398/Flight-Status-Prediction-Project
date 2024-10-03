from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

df = pd.read_csv('Combined_Flights_2022.csv')

#label encoding
le_airline = LabelEncoder()
df['Airline'] = le_airline.fit_transform(df['Airline'])

le_op_airline = LabelEncoder()
df['Operating_Airline'] = le_op_airline.fit_transform(df['Operating_Airline'])

le_origin = LabelEncoder()
df['Origin'] = le_origin.fit_transform(df['Origin'])

le_dest = LabelEncoder()    
df['Dest'] = le_dest.fit_transform(df['Dest'])


app = Flask(__name__)
#load the model
model = pickle.load(open('flightmodel.sav', 'rb'))


@app.route('/')
def home():
    result = ' '
    return render_template('index.html', **locals())

@app.route('/predict', methods=['POST'])
def predict():
    CRSDepTime = int(request.form['CRSDepTime'])
    CRSArrTime = int(request.form['CRSArrTime'])
    arrTime = int(request.form['arrTime'])
    CRSElapsedTime = int(request.form['CRSElapsedTime'])
    month = int(request.form['month'])
    dayOfWeek = int(request.form['dayOfWeek'])
    day = int(request.form['day'])
    airline = request.form['airline']
    op_airline = request.form['op_airline']
    origin = request.form['origin']
    dest = request.form['dest']

    input_df = pd.DataFrame([[CRSDepTime, CRSArrTime, arrTime, CRSElapsedTime, month, 
                              dayOfWeek, day, airline, op_airline, origin, dest]], 
                              columns=['CRSDepTime', 'CRSArrTime', 'ArrTime', 'CRSElapsedTime', 
                                     'Month', 'DayOfWeek', 'Day', 'Airline', 'Operating_Airline', 'Origin', 'Dest'])
    
    input_df['Airline'] = le_airline.transform(input_df['Airline'])
    input_df['Operating_Airline'] = le_op_airline.transform(input_df['Operating_Airline'])
    input_df['Origin'] = le_origin.transform(input_df['Origin'])
    input_df['Dest'] = le_dest.transform(input_df['Dest'])

    x = input_df.iloc[:, :11].values
    prediction = model.predict(x)

    if(prediction[0] == 0):
       result = "The flight is not cancelled"
    else:
       result = "The flight is cancelled"

    return render_template('index.html', **locals())    

if __name__ == '__main__':
    app.run(debug=True)