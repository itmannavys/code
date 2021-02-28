import joblib
import numpy as np
from flask import Flask, render_template, request


model = joblib.load('iris.pkl')
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    petal_length = request.form['petal_length']
    petal_width = request.form['petal_width']
    arr = np.array([[petal_length, petal_width]])
    predict_ = model.predict(arr)
    return render_template('result.html', data=predict_)


@app.route('/figure')
def figure():
    pass


if __name__ == "__main__":
    app.run(debug=True)
