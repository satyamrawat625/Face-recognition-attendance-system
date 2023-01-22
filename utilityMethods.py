import os
from datetime import date
from datetime import datetime
import pandas as pd
import joblib


#### Saving Date today in 2 different formats
def datetoday():
    return date.today().strftime("%m_%d_%y")


def datetoday2():
    return date.today().strftime("%d-%B-%Y")


#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


#### Identify face using ML model
def identify_face(facearray):#chng
    model = joblib.load('static/EncodeFile.pkl')
    return model.predict(facearray)


#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday()}.csv')
    ID = df['ID']
    names = df['Name']
    times = df['Time']
    l = len(df)
    return  ID,names, times, l


#### Add Attendance of a specific user
def add_attendance(name):
    userid = name.split('_')[0]
    username = name.split('_')[1]

    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday()}.csv')
    if int(userid) not in list(df['ID']):
        with open(f'Attendance/Attendance-{datetoday()}.csv', 'a') as f:
            f.write(f'\n{userid},{username},{current_time}')

