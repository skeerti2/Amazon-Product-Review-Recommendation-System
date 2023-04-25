""""
Logistic Regression model for amazon book dataset
Submitted by : Gmon Kuzhiyanikkal
Date : 23 april 2023

"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


# Read the TSV file and select only the first 1000 rows
df = pd.read_csv("book.tsv", delimiter="\t", nrows=1000)

# Create a new column "sentiment" based on the "star_rating" column
df['sentiment'] = df['star_rating'].apply(lambda x: 'positive' if x>=4 else ('neutral' if x==3 else 'negative'))

# Preprocess the "review_body" column using tokenization, stop word removal, and stemming
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

def preprocess(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Perform stemming
    stemmed_tokens = [porter.stem(token) for token in filtered_tokens]
    # Join the tokens back into a string
    return " ".join(stemmed_tokens)

df['review_body'] = df['review_body'].apply(preprocess)

# Split the data into training and testing sets
y = df['sentiment']
X = df['review_body']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for logistic regression with CountVectorizer and TfidfTransformer
clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('lr', LogisticRegression(random_state=42))
])

# Train the logistic regression classifier on the training set
clf.fit(X_train, y_train)

# Evaluate the performance of the classifier on the testing set
print("\nlogistic regression classifier\n")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 score:", f1)
