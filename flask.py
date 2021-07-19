
import numpy as np
from flask import Flask,request,jsonify,render_template
import  pandas as pd
import joblib,pickle

app = Flask(__name__)
model = joblib.load('hotel.pkl')
sc = joblib.load('hotel_scalar.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    x_col = ['name',  'Check_in_year', 'Check_in_month', 'Check_in_day', 'Check_out_year', 'Check_out_month',
       'Check_out_day', 'city']
    
    data = np.array([[x for x in request.form.values()]])
    data = sc.transform(data)
    
    prediction = model.predict(data)
    
    print(prediction)
    return render_template('index.html',prediction_text=prediction)

if __name__ == '__main__':
    app.run(debug=True)
    
    
