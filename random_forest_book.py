""""
Random Forest model for amazon book dataset
Submitted by : Gmon Kuzhiyanikkal
Date : 23 april 2023

"""


import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# Read the TSV file and select only the first 1000 rows
df = pd.read_csv("book.tsv", delimiter="\t", nrows=10000)

# Create a new column "sentiment" based on the "star_rating" column
df['sentiment'] = df['star_rating'].apply(lambda x: 'positive' if x>=4 else ('neutral' if x==3 else 'negative'))

# Tokenize, remove stop words, and stem the "review_body" column using the NLTK library
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def tokenize(text):
    tokens = word_tokenize(text.lower())
    return [stemmer.stem(word) for word in tokens if word not in stop_words]

vectorizer = CountVectorizer(tokenizer=tokenize)
X = vectorizer.fit_transform(df['review_body'])

# Split the data into training and testing sets
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier on the training set
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate the performance of the classifier on the testing set
print("\nrandom forest classifierr\n")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

y_pred = clf.predict(X_test)
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 score:", f1)
