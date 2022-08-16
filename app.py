import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import datetime

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
features = []


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    date_format = datetime.datetime.strptime(features[-1], "%Y-%m-%d")
    features[-1] = datetime.datetime.timestamp(date_format)
    final_features = [np.array(features).astype(float)]
    print(final_features)
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index.html', prediction_text='The probability that this transaction is fraudulent is {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
