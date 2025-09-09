from flask import Flask, request
from model import generate
import pickle

generate()

ai = pickle.load(open('ai.pkl', 'rb'))

app = Flask(_name_)

@app.route('/')
def homepage():
    return "Server Runnionggggggggggggggggggg"

@app.route('/predict')
def predict():
    ir = request.args.get('ir')
    ir = int(ir)
    data = [[ir]]
    result = ai.predict(data)[0]
    return result


if (_name) == "main_":

    app.run(host = '0.0.0.0',port=4000)
