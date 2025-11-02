import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

def convert_score_to_label(score):
    """
    Converts a sentiment score to a sentiment label.
    """
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"

def train_naive_bayes(csv_file):
    """
    Trains and evaluates a Naive Bayes classifier on text data from a CSV file.

    Args:
        csv_file (str): Path to the CSV file containing news articles.
    """
    # Load data
    data = pd.read_csv(csv_file, encoding='utf-8')

    # Extract text and convert sentiment scores to labels
    X = data.iloc[:, 2]  # Text column
    raw_scores = data.iloc[:, 3]  # Continuous sentiment scores
    y = raw_scores.apply(convert_score_to_label)

    # Show class distribution
    print("Class distribution:")
    print(y.value_counts(), "\n")

    # Vectorize text
    vectorizer = CountVectorizer(stop_words='english')
    X_vec = vectorizer.fit_transform(X)

    # Convert to TF-IDF
    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X_vec)

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42
    )

    # Train Multinomial Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Evaluate
    print("=== Evaluation Results ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, average='weighted', zero_division=0))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted', zero_division=0))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=["negative", "neutral", "positive"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["negative", "neutral", "positive"])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    # Save model and vectorizer
    pickle.dump(vectorizer, open('vectorizer_crude_oil.pkl', 'wb'))
    pickle.dump(clf, open('naive_bayes_classifier_crude_oil.pkl', 'wb'))

if __name__ == '__main__':
    csv_file = '/Users/karos/Documents/Quantitative Trading/Sentiment Analysis/Machine Learning/CrudeOil_News_Articles.csv'
    train_naive_bayes(csv_file)
