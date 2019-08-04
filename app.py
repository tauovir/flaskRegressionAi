from flask import Flask, redirect, url_for,request,render_template
import pandas as pd
import numpy as np
from sklearn.externals import joblib
app = Flask(__name__)
app.templates_auto_reload = True
#===================Login Funda=============
@app.route('/')
def investmentForm():
     return render_template('formpage.html')
    
@app.route('/predict',methods = ['POST', 'GET'])
def prediction():
    if request.method == 'POST':
        try:
            dictVales = arrangeData(request)
            regressor = open("linearTrainedModel.pkl","rb")
            regressorModel = joblib.load(regressor)
            predictVal = regressorModel.predict(dictVales)
        except valueError:
            return "error occure"
    return render_template('predict.html', profit = round(predictVal[0],2))     

def arrangeData(request):
        rnd = float(request.form['rnd'])
        admin = float(request.form['admin'])
        marketing = float(request.form['marketing'])
        state = request.form['state']
        # We need labelEncodet and onHot Encoder to transform categorical data in same pattern which was used at model training time
        labelEncoder = open("labelEncoderObject.pkl","rb")
        oneHotEncoder = open("oneHoteEncoderObject.pkl","rb")
        labelEncoderObj = joblib.load(labelEncoder)
        oneHotEncoderObj = joblib.load(oneHotEncoder)
        inputData = np.array([rnd,admin,marketing,state]).reshape(1,-1)
        inputData[:,3] = labelEncoderObj.transform(inputData[:,3])
        inputData = oneHotEncoderObj.transform(inputData).toarray()
        return inputData
     
@app.route('/login',methods = ['POST', 'GET'])
def login():
    if request.method == 'POST':
        user = request.form['nm']
        return redirect(url_for('success', name = user))
@app.route('/form',methods = ['get'])
def test():
    return "khan is here"
app.run()



#=====================
