import json
import numpy as np
import os
import pickle
import traceback
import pandas as pd
from sklearn.externals import joblib

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'automl_model.pkl')
    model = joblib.load(model_path)
    #with open(model_path, 'rb') as f:
     #   model = pickle.load(f)

def run(data):
    try:
        data = np.array(json.loads(data))
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        return str(e)