from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
import pickle
from sklearn.feature_extraction.text  import TfidfVectorizer
import pandas as pd
app = Flask(__name__)

with open('flask_app/vectorizer.pkl', 'rb') as f:
    vect = pickle.load(f)
model = pickle.load(open('flask_app/model.plk', 'rb'))


def predict_genre(overview):
    data = []
    overview = tokenize(overview)
    data.append(overview)
    df = pd.DataFrame(data, columns=['overview'])
    print(data)
    vect = TfidfVectorizer(max_features=1500, ngram_range=(1, 15))
    vect_features = vect.transform([data])
    tfidf_df = pd.DataFrame(vect_features.toarray(), columns=vect.get_feature_names_out())
    genre = model.predict(tfidf_df)
    return genre

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
