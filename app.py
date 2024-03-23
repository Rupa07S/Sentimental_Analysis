import joblib 
import pandas as pd
import re
from flask import Flask
import contractions
from nltk.stem import WordNetLemmatizer
from flask import render_template,request
from sklearn.feature_extraction.text import CountVectorizer

model = joblib.load(r"C:\Users\swaro\OneDrive\Desktop\Internship assignment-flipkart\reviews_badminton\sentimental_analysis.pkl")


def text_preprocessing(text):

    text = text.lower()
    text = text.replace("read more","")

    text = " ".join([contractions.fix(word) for word in text.split()])

    text = re.sub("[^a-zA-Z.]"," ",text)

    text = " ".join([WordNetLemmatizer().lemmatize(word) for word in text.split()])
    
    return text

def vectorize(review):
   clf =  CountVectorizer()
   test = clf.transform(review.split())
   return test




app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/senti",methods=["POST","GET"])
def senti():
    review = request.form.get("review")
    text = text_preprocessing(review)
    text = vectorize(text)

    if model.predict(text) == 0:
        return render_template("index.html",sentiment="Negative")
    else:
        return render_template("index.html",sentiment="Positive")
    
   
 

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port="5000")