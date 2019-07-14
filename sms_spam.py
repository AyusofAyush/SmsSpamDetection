# Combined Efforts of Ayush, Rahul and Shalini
# NLP and Naive Bayes Classification used.

#importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('SMSCollection.txt', delimiter = '\t', quoting = 3)

#cleaning text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0,1536):
    review = re.sub('[^a-zA-Z]', ' ', dataset['review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#Creating Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()    
y = dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

i = pd.read_csv('sample.txt', delimiter = '\t', quoting = 3)

lst = []
r = re.sub('[^a-zA-Z]', ' ', i['Review'][0])
r = r.lower()
r = r.split()
ps = PorterStemmer()
r = [ps.stem(word) for word in r if not word in set(stopwords.words('english'))]
r = ' '.join(r)
lst.append(r)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 100)
m = cv.fit_transform(lst).toarray() 

y_p = classifier.predict(m)

print(y_pred)