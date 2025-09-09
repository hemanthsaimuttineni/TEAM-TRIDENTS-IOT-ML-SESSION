from flask import Flask, request
from model import generateAi
import pickle

# Generate model if needed
generateAi()

# Load the trained model
ai = pickle.load(open('ai.pkl', 'rb'))

# Create Flask app
app = Flask(__name__)

@app.route('/')
def homepage():
    return "Server is running"

@app.route("/predict")
def predict():
    # Get input parameter (ir)
    ir = request.args.get('ir')
    if ir is None:
        return "Error: Missing 'ir' parameter", 400
    
    try:
        ir = int(ir)
    except ValueError:
        return "Error: 'ir' must be an integer", 400
    
    # Prepare data
    data = [[ir]]
    result = ai.predict(data)[0]   # Prediction
    
    return str(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4000, debug=True)
