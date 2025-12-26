from flask import Flask,request,jsonify,render_template

# render_template for finding the html port
# jsonify getting data from website
# request for getting data from the client

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import ridge regressor and standard scaler pickle
ridge_model=pickle.load(open(r'C:\Users\abhip\Desktop\projects\Algerian-forest-fire-prediction\model\ridge_model.pkl','rb'))
scaler=pickle.load(open(r'C:\Users\abhip\Desktop\projects\Algerian-forest-fire-prediction\model\scaler.pkl','rb'))
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predictdata',methods=['POST','GET'])
## GET --> we are retrieving and POST ---> we are sending some data
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_data_scaled=scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)
        return render_template('home.html',result=result[0])
    else:
        return render_template("home.html")

if __name__=="__main__":
    app.run(host="0.0.0.0")