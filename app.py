# Import the Flask class from the flask module
from flask import Flask, render_template, request
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


# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)