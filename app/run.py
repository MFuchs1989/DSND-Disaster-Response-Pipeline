import joblib
import json
import re
import pandas as pd
from sqlalchemy import create_engine

import plotly
from plotly.graph_objs import Bar, Histogram

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from flask import Flask
from flask import request, jsonify, render_template


# Create app
app = Flask(__name__)


# List of stopwords
stop = stopwords.words('english')


def get_top_words(txt, num_words=10):

    '''
    Find words with the highest document frequency.

    Args:
        txt (list-like object): text data
        num_words (int): number of words to find

    Returns:
        (Pandas series) top words and their document frequencies
    '''

    tfidf = TfidfVectorizer(stop_words=stop, max_features=num_words)
    tfidf.fit(txt)
    words = tfidf.vocabulary_
    
    for word in words:
        words[word] = txt[txt.str.contains(word)].count()
    return pd.Series(words).sort_values()


def tokenize(text):

    '''
    Tokenizes message data

    Args:
        text (str): Text to tokenize

    Returns:
        (DataFrame) clean_messages: array of tokenized message data
    '''

    #Case Normalization & remove punctuation 
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) 
    
    #tokenization methods 
    
    words = word_tokenize(text)
    
    words = [w for w in words if w not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]

    
    return words


# Load data
engine = create_engine('sqlite:///../data/messages.db')
df = pd.read_sql_table('messages', engine)


# Load model
model = joblib.load("../models/classifier.pkl")


# Home page
@app.route('/')
@app.route('/index')
def index():

    # Category counts
    cat_counts = df.iloc[:, 4:].sum().sort_values()[-10:]
    cat_names = cat_counts.index.tolist()

    # Message word counts
    word_counts = df.message.apply(lambda s: len(s.split()))
    word_counts = word_counts[word_counts <= 100]

    # Genre counts
    genre_counts = df.groupby('genre').count()['message']
    genre_names = genre_counts.index.tolist()

    # Top word
    top_counts = get_top_words(df.message)
    top_words = top_counts.index.tolist()

    # Create Visualizations
    graphs = [

        # Category counts
        {
            'data': [
                Bar(
                    x=cat_counts,
                    y=cat_names,
                    orientation='h'
                )
            ],
            'layout': {
                'title': 'Top Message Categories',
                'yaxis': {'title': "Category"},
                'xaxis': {'title': "Number of Messages"},
                'margin': {'l': 100}
            }
        },

        # Top word counts
        {
            'data': [
                Bar(
                    x=top_counts,
                    y=top_words,
                    orientation='h'
                )
            ],
            'layout': {
                'title': 'Most Common Words in Messages',
                'yaxis': {'title': "Word"},
                'xaxis': {'title': "Number of Messages"},
                'margin': {'l': 100}
            }
        },

        # Message word counts
        {
            'data': [
                Histogram(
                    x=word_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Word Counts',
                'yaxis': {'title': "Number of messages"},
                'xaxis': {'title': "Word count"}
            }
        },    
        
        # Genre counts
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Number of messages"},
                'xaxis': {'title': "Genre"}
            }
        }    
    ]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()








