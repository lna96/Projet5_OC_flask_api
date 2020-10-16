# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 19:56:59 2020

@author: linaj
"""

import pickle
from flask import Flask, request, render_template
import nltk
from nltk.stem import WordNetLemmatizer


app = Flask(__name__)

#app.config.from_object(os.environ['APP_SETTINGS'])
#app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

QuerySO_model = pickle.load(open('model.pkl','rb'))
tf_transformer = pickle.load(open('tf_transformer.pkl','rb'))
le = pickle.load(open('le.pkl','rb')) #LabelEncoder
      
@app.route('/', methods=['POST', 'GET'])
def predict():
    tags_pred = []
    if request.method == "POST":
        text = request.form['text']
 
        # Tokenisation
        tokenizer = nltk.RegexpTokenizer(r'\w+')
        text_tk = tokenizer.tokenize(text.lower())
        # Remove numbers, but not words that contain numbers.
        text_tk = [word for word in text_tk if not word.isnumeric()] 
        # Lemmatisation
        lemmatizer = WordNetLemmatizer()
        text_tk = [lemmatizer.lemmatize(word) for word in text_tk if nltk.pos_tag([word])[0][1][0]=='N']
        # TfidfVectorizer
        text_tk = ' '.join([word for word in text_tk])
        text_tfidf = tf_transformer.transform([text_tk])
        # predict tag
        y_pred = QuerySO_model.predict(text_tfidf)
        tags_pred = list(le.inverse_transform(y_pred))
    
    return render_template('index.html', tags_pred=tags_pred)
    



if __name__ == "__main__": 
    app.run(debug=True)
    