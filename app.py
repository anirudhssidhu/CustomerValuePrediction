import pickle
from flask import (Flask,request,app,jsonify,url_for,render_template)

import numpy as np
import pandas as pd

app=Flask(__name__)

# loading the model
regmodel=pickle.load(open("./models/bestMODEL.pkl", "rb"))
dropCols = pickle.load(open("./models/dropCol.pkl", "rb"))
impCols = pickle.load(open("./models/impCol.pkl", "rb"))
colNames = pickle.load(open("./models/colNames.pkl", "rb"))

@app.route('/')
def home():
  return render_template('home.html')

@app.route('/predict', methods = [ 'POST'])
def data():
    if request.method == 'POST':
        f = request.form['csvfile']
        with open(f) as file:
            df = pd.read_csv(file)    

        # preprocessing the input 
        df.drop(dropCols,axis=1,inplace=True)
        df = df[impCols]

        input = pd.DataFrame(np.log1p(df),columns=impCols)

        # predicting the output
        y_hat = regmodel.predict(input)
        output = round(np.expm1(y_hat[0]),2)

        # prediction message
        predictionText = f"The Customer Value is = {output}"

        return render_template("home.html",predictionText=predictionText)


if __name__ == "__main__":
    app.run(debug=True)