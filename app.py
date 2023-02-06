import pickle
from flask import (Flask,request,app,jsonify,url_for,render_template)

import numpy as np
import pandas as pd

app=Flask(__name__)

# loading the model
regmodel=pickle.load(open("./models/best_model.pkl", "rb"))

@app.route('/')
def home():
  return render_template('home.html')

@app.route('/predict_api', methods=[ 'POST',] )
def predict_api():
  