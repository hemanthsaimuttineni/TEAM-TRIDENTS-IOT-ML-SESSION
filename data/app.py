from flask import flask
from model.py import generateAi
import pickle

generateAi()
ai=pickle.load(open('ai.pkl','rb'))
app=flask(__name__)

@app.route('/')
def homepage():
    return "server running"

@app.route("/predict")#/predict?ir=0
def predict():
    ir=request.args.get('ir')
    ir=int(ir)
    data=[[ir]]
    result=ai.predict(data)[0]#["object"]
    return result

if (__name__)=="__main__":
    app.run(host='0.0.0.0',port=8080)