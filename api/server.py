import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, json
import dill as pickle

app = Flask(__name__)

@app.route("/predict", methods = ["POST"])
def apicall():

    try:
        json_data = json.loads(request.data)
        data = pd.DataFrame()
        data["Дата"] = pd.date_range(pd.to_datetime(json_data["startDate"]), pd.to_datetime(json_data["endDate"])-pd.Timedelta(days = 1), freq = "d")
        data["Товар"] = json_data["product"]
        data["Склад"] = json_data["warehouseId"]
    except Exception as e:
        raise e
    
    regressor = "model_v1.pk"

    if data.empty:
        return bad_request()
    else:
        print("Loading model.")
        model = None
        with open("./models/" + regressor, "rb") as file:
            model = pickle.load(file)
        print("Model loaded.")
        print("Making prediction.")
        predictions = model.predict(data)
        prediction = np.sum(predictions)
        
        print("Send prediction.")
        response = jsonify(prediction = prediction)
        response.status_code = 200

        return response
        

@app.errorhandler(400)
def bad_request(error=None):
	message = {
			'status': 400,
			'message': 'Bad Request: ' + request.url + '--> Please check your data payload...',
	}
	resp = jsonify(message)
	resp.status_code = 400

	return resp