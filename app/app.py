import os
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
import sys
import pandas as pd

PROJECT_ROOT = Path(os.path.abspath('')).resolve()

#sys.path.insert(0, os.path.join(PROJECT_ROOT,'data','radar.py'))
#sys.path.insert(1, os.path.join(PROJECT_ROOT,'data'))
#from myfolder.myfile import myfunc

app = Flask(__name__)  # Initialize the flask App
model = load(os.path.join(PROJECT_ROOT, 'model', 'best_rf2.joblib'))
scaler = load(os.path.join(PROJECT_ROOT, 'model', 'scaler2.joblib'))
encoder = load(os.path.join(PROJECT_ROOT, 'model', 'encoder2.joblib'))


@app.route('/')
def home():
    return render_template('Home.html', prediction_text= "PLEASE INSERT BOOKING DATA", prediction = 999)

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features_name = ['LeadTime', 'ArrivalDateMonth', 'ArrivalDateWeekNumber',
       'ArrivalDateDayOfMonth', 'StaysInWeekendNights', 'StaysInWeekNights',
       'Adults', 'Children', 'Babies', 'Meal', 'MarketSegment',
       'DistributionChannel', 'IsRepeatedGuest', 'PreviousCancellations',
       'PreviousBookingsNotCanceled', 'ReservedRoomType', 'AssignedRoomType',
       'BookingChanges', 'Agent', 'Company', 'DaysInWaitingList',
       'CustomerType', 'ADR', 'RequiredCarParkingSpaces',
       'TotalOfSpecialRequests']
    # To get data from input form in the web
    int_features = list(request.form.values())[0].split(',')
    # Reshape the input values to desired array
    final_features = np.array(int_features).reshape(1, -1)
    # Convert to dataframe with all features name
    final_features = pd.DataFrame(final_features,columns=features_name)
    num= ~final_features.columns.isin(encoder.cols)
    final_features[final_features.columns[num]] = final_features[final_features.columns[num]].astype(float)
    # Encode the data
    final_features[encoder.cols] =  encoder.transform(final_features[encoder.cols])
    # Scale the data
    final_features = scaler.transform(final_features) 
    # Perform prediction
    prediction = model.predict(final_features)[0]
    prediction_proba = model.predict_proba(final_features)[0][1]
    if prediction == 1:
        if prediction_proba < 0.8:
            profile = "BE CAREFUL !!!"
            prediction_text_detail = ["The customer has a probability of " + '{0:.2f}'.format(prediction_proba* 100) + "% to cancel.",
                                      "Advisable to apply business action to avoid the cancellation.",
                                      "Examples : free parking, discount for children, free meal, etc."]
        else:
            profile = "ACTION NEEDED !!!"
            prediction_text_detail = ["The customer has a probability of " + '{0:.2f}'.format(prediction_proba* 100) + "% to cancel.",
                                      "Many other factors could affect the booking",
                                      "Advisable to consider the booking canceled, and eventually apply a restrict deposit policy or overbook the room"]
    elif prediction == 0:
        profile = "THIS BOOKING IS GOOD"
        prediction_text_detail = ["The customer has a probability of " + '{0:.2f}'.format(prediction_proba* 100) + "% to cancel.",
                                      "No more actions required"]

    else:
        profile = "ERROR IN DATA!"


    return render_template('Home.html', 
            prediction_text= profile,
            prediction = prediction,
            prediction_proba = prediction_proba,
            prediction_text_detail = prediction_text_detail
            )

if __name__ == "__main__":
    app.run(use_debugger=False, use_reloader=False, passthrough_errors=True)





