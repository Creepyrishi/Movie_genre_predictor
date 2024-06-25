from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
import pickle
from sklearn.feature_extraction.text  import TfidfVectorizer
import pandas as pd
app = Flask(__name__)

vect = pickle.load(open('flask_app/vect.pkl', 'rb'))
model = pickle.load(open('flask_app/model.pkl', 'rb'))
ohe = pickle.load(open('flask_app/ohe.pkl', 'rb'))

def predict_genre(overview):
    data = []
    overview = tokenize(overview)
    data.append(overview)
    df = pd.DataFrame(data, columns=['overview'])
    vect_features = vect.transform(df['overview'])
    prediction = model.predict(vect_features)
    genre = ohe.inverse_transform(prediction)[0]
    return genre[0]

def tokenize(text):
    stop_words = set(stopwords.words('english'))
    text_words = word_tokenize(text, 'english')
    text_token = [tokens for tokens in text_words if tokens.lower() not in stop_words and  tokens.lower() not in punctuation]
    return (' '.join(text_token)).lower()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        overview = request.form['overview']
        genre = predict_genre(overview)
        return render_template('index.html', overview=overview, genre=genre)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)