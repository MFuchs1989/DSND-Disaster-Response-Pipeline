import sys
import joblib
import re
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# List of stopwords
stop = stopwords.words('english')


def load_data(database_filepath):

    '''
    Load in the clean dataset from the SQLite database.

    Args:
        database_filepath (str): path to the SQLite database

    Returns:
        (DataFrame) X: Independent Variables , array which contains the text messages
        (DataFrame) Y: Dependent Variables , array which contains the labels to the messages
        (DataFrame) categories: Data Column Labels , a list with the target column names, i.e. the category names
    '''

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM messages', engine)
    X = df['message'].copy()
    Y = df.loc[:, 'related':'direct_report']
    categories = Y.columns.tolist()
    return X, Y, categories



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


def build_model():

    '''
    Build a machine learning pipeline that converts text data into a numeric vector then classifies multiple binary
    target labels.

    Args:
        None

    Returns:
        (Sklearn pipeline) pipeline estimator
    '''

    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

     # Declaring parameters
    parameters = {'clf__estimator__min_samples_leaf': [2, 3, 4]}

    
    cv = GridSearchCV(pipeline, parameters)
    
    return cv   




def evaluate_model(model, X_test, Y_test, category_names):

    '''
    Evaluate the machine learning model using a test dataset and print the classification report metrics for each label.

    Args:
        model (Sklearn estimator): machine learning model
        X_test (list-like object): test set text data
        Y_test (Pandas dataframe): test set target labels
        category_names (list): names of target labels

    Returns:
        (Pandas dataframe) Classification report metrics
    '''

    pred = pd.DataFrame(model.predict(X_test), columns=category_names)

    metrics = []

    for col in category_names:
        report = classification_report(Y_test[col], pred[col])
        scores = report.split('accuracy')[1].split()
        metrics.append([float(scores[i]) for i in [0, 4, 5, 6, 10, 11, 12]])

    metric_names = ['accuracy', 'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1', 'weighted_avg_precision',
                    'weighted_avg_recall', 'weighted_avg_f1']
    metrics_df = pd.DataFrame(metrics, columns=metric_names, index=category_names)

    print(metrics_df)
    print(metrics_df.sum)
    return metrics_df
        

def save_model(model, model_filepath):

    '''
    Save the machine learning model as a pickle file.

    Args:
        model (Sklearn estimator): machine learning model
        model_filepath (str): path to save the model

    Returns:
        None
    '''

    joblib.dump(model, model_filepath)
    return


def main():

    '''
    This file is the ML pipeline that trains the classifier and saves it as a pickle file.

    From this project's root directory, run this file with:
    python models/train.py data/messages.db models/classifier.pkl
    '''

    if len(sys.argv) == 3:

        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:

        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train.py ../data/messages.db classifier.pkl')


if __name__ == '__main__':
    main()

