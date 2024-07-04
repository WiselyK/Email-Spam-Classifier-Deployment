# Import the Flask class from the flask module
from flask import Flask, render_template, request, jsonify
import pickle

# Create an instance of the Flask class
app = Flask(__name__)
cv = pickle.load(open("models/cv.pkl", "rb")) #Count vectorizer/tokenizer
clf = pickle.load(open("models/clf.pkl", "rb")) #Classifier

# Register a route
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email = request.form.get('content')
    tokenized_email = cv.transform([email]) # X 
    prediction = clf.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1
    return render_template("index.html", prediction=prediction, email=email)

#Create an API route
@app.route('/api/predict', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)  # Get data posted as a json
    email = data['content']
    tokenized_email = cv.transform([email]) # X 
    prediction = clf.predict(tokenized_email)
    # If the email is spam prediction should be 1
    prediction = 1 if prediction == 1 else -1
    return jsonify({'prediction': prediction, 'email': email})  # Return prediction

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)