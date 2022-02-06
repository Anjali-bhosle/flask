
import numpy as np
from flask import Flask,request,jsonify,render_template
import os
import  pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('xgb_regressor.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    x_col = ['name', 'city', 'Distance to City centre(km)',
       'Distance to Airport(km)', 'Pool', 'Free parking', 'Airport transfer',
       'Spa', 'Restaurant', 'Gym', 'Bar', 'Bathtub', 'Meeting Facilities',
       'Connecting rooms available', 'Pet-friendly', 'Kitchen',
       'Internet access', 'Check_in_year', 'Check_in_month', 'Check_in_day',
       'Check_out_year', 'Check_out_month', 'Check_out_day']
    
    data = [[x for x in request.form.values()]]
    print(data)
    
    data = pd.DataFrame(data,columns=x_col)
    
    prediction = model.predict(data)
    
    print(prediction)
    return render_template('index.html',prediction_text=prediction)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
