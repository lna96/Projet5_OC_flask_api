# train model 
#import sklearn.external.joblib as extjoblib
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

def train(df):
    
    df.dropna(inplace=True)
    tf_transformer = TfidfVectorizer(min_df=5).fit(df.Title)
    X_counts_tf = tf_transformer.transform(df.Texts)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(df.Tags)
    
    X_train,X_test,y_train,y_test = train_test_split(X_counts_tf, y, test_size=0.3, random_state=0)

    clf_svc =  OneVsRestClassifier(LinearSVC(C=0.01, dual=False))
    clf_svc.fit(X_train, y_train)
    
    return [clf_svc, tf_transformer, le]

if __name__ == '__main__':

    df = pd.read_csv(r'C:\Users\linaj\Desktop\Openclassroom\P5\StackOverflow_data.csv', sep=';')

    model = train(df)[0]
    tf_transformer = train(df)[1]
    le = train(df)[2]

    # Saving model to disk
    pickle.dump(model, open('model.pkl','wb'))
    pickle.dump(tf_transformer, open('tf_transformer.pkl','wb'))
    pickle.dump(le, open('le.pkl','wb'))
    
        
