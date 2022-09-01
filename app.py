import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import os

port = int(os.environ.get('PORT', 5000))
app=Flask(__name__)

##Load the model (Model yükleme)
model=pickle.load(open("completeMLRegressionProject/regmodel.pkl","rb"))

@app.route("/")

def home():
    return render_template("home.html")



@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    print(data)
    data=[np.array(data)]
    output=model.predict(data)
    return render_template("home.html",prediction_text="Boston City House Price Prediction is {}".format(output))



if __name__=="__main__":
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=port, debug=True)