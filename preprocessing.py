# ==========================================
# Sentiment Analysis using NLP
# ==========================================

import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK data
nltk.download('stopwords')

# Load CSV File
df = pd.read_csv("C:/Users/Harshali_Patil/Documents/Natural Language Processing/Sentiment_Analysis_Project/reviews.csv") 
print("\nDataset Loaded Successfully")
print("Total Reviews:", len(df))

print("\nSentiment Count:")
print(df['Sentiment'].value_counts())

# Text Preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Clean Reviews
df['Clean_Review'] = df['Review'].apply(clean_text)

# Features and Labels
X = df['Clean_Review']
y = df['Sentiment']

# Convert text into numbers
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X)

# Train Model
model = LogisticRegression(max_iter=1000)

model.fit(X, y)

print("\nModel Training Completed")

# Prediction Loop
while True:

    review = input("\nEnter a review (or type exit): ")

    if review.lower() == "exit":
        print("Program Ended")
        break

    cleaned_review = clean_text(review)

    review_vector = vectorizer.transform([cleaned_review])

    prediction = model.predict(review_vector)

    probability = model.predict_proba(review_vector)

    confidence = round(max(probability[0]) * 100, 2)

    print("\nOriginal Review :", review)
    print("Cleaned Review  :", cleaned_review)
    print("Sentiment       :", prediction[0])
    print("Confidence      :", confidence, "%")