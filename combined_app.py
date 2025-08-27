from flask import Flask, render_template, request, redirect, url_for
import math

# --- Imports for Task 1 (Search) ---
from ranker import load_index, search_publications_tfidf

# --- Imports for Task 2 (News Classifier) ---
import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.calibration import CalibratedClassifierCV

# --- Setup ---
import warnings
warnings.filterwarnings('ignore')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

app = Flask(__name__, template_folder='templates')

# --- Task 1: Search Interface ---
vectorizer, tfidf_matrix, publications = load_index()

@app.route("/", methods=["GET", "POST"])
def home():
    query = request.form.get("query") or request.args.get("query")
    page = int(request.args.get("page", 1))
    per_page = 10
    results = []
    if query:
        results = search_publications_tfidf(query, vectorizer, tfidf_matrix, publications)
    total_results = len(results)
    total_pages = math.ceil(total_results / per_page)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_results = results[start:end]
    return render_template(
        "index.html",
        query=query,
        results=paginated_results,
        page=page,
        total_pages=total_pages,
        total_results=total_results
    )

# --- Task 2: Text Classifier Interface ---
class DocumentClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(sublinear_tf=True)),
            ('select', SelectKBest(chi2)),
            ('clf', CalibratedClassifierCV(
                LinearSVC(class_weight='balanced', max_iter=10000, dual=False),
                cv=3
            ))
        ])
        self.best_estimator_ = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        if not isinstance(text, str):
            return ""
        tokens = word_tokenize(text.lower())
        tokens = [w for w in tokens if w.isalpha() and w not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(w) for w in tokens]
        return " ".join(tokens)

    def load_data(self, csv_file):
        df = pd.read_csv(csv_file)
        df['text_to_process'] = df['title'].fillna('') + " " + df['summary'].fillna('')
        df['processed_text'] = df['text_to_process'].apply(self.preprocess)
        df.dropna(subset=['category'], inplace=True)
        return df

    def train(self, X, y, tune=True):
        if tune:
            param_grid = {
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'tfidf__max_df': [0.85, 0.95],
                'select__k': [3000, 5000, 'all'],
                'clf__estimator__C': [0.1, 1, 10]
            }
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid = GridSearchCV(self.pipeline, param_grid, scoring='f1_macro', cv=cv, n_jobs=-1)
            grid.fit(X, y)
            self.best_estimator_ = grid.best_estimator_
        else:
            self.pipeline.fit(X, y)
            self.best_estimator_ = self.pipeline

    def predict(self, text):
        if self.best_estimator_ is None:
            raise RuntimeError("Model not trained yet.")
        processed = self.preprocess(text)
        probs = self.best_estimator_.predict_proba([processed])[0]
        idx = np.argmax(probs)
        return self.best_estimator_.classes_[idx], probs[idx]

# Load and train classifier once
classifier = DocumentClassifier()
df = classifier.load_data('news_300_dataset.csv')
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], df['category'], test_size=0.2, random_state=42, stratify=df['category']
)
classifier.train(X_train, y_train, tune=True)

# Print and plot confusion matrix and classification report for test set
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

y_pred = [classifier.predict(text)[0] for text in X_test]
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
print(f"Model trained.\nOverall Test Accuracy: {acc:.4f},\nMacro F1 Score: {f1:.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred))
print("Confusion Matrix (Test Set):")
cm_test = confusion_matrix(y_test, y_pred)
print(cm_test)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens',
            xticklabels=classifier.best_estimator_.classes_,
            yticklabels=classifier.best_estimator_.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Test Set)')
plt.tight_layout()
plt.show()

history = []

@app.route("/task2", methods=["GET", "POST"])
def task2():
    global history
    result = None
    confidence = None
    user_input = ""
    if request.method == "POST":
        user_input = request.form.get("headline", "").strip()
        if user_input:
            result, confidence = classifier.predict(user_input)
            confidence = round(confidence * 100, 2)
            history.append((user_input, result, confidence))
            history = history[-5:]  # keep only last 5
    return render_template("news_index.html", result=result, confidence=confidence, user_input=user_input, history=history)

if __name__ == "__main__":
    app.run(debug=True)