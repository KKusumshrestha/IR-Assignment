# indexer.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json

def load_data(csv_file):
    """Loads publication data from a CSV file."""
    try:
        df = pd.read_csv(csv_file)
        #json.loads to parse the string into a list of dictionaries
        df['authors'] = df['authors'].apply(
            lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else []
        )
        publications = df.to_dict('records')
        return publications
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found. Please run crawler.py first.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in 'authors' column: {e}")
        return df.copy()

def build_tfidf_index(publications):
    """
    Builds a TF-IDF matrix for all publications and saves the model.
    """
    if not publications:
        return None, None, None
        
    # Combine fields into one text per publication
    docs = []
    for pub in publications:
        authors_text = ' '.join([a['name'] for a in pub['authors']])
        document_text = f"{pub['title']} {authors_text} {pub['date']} {pub.get('abstract', '')}"
        docs.append(document_text)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(docs)

    return vectorizer, tfidf_matrix, publications

def save_index(vectorizer, tfidf_matrix, publications, filename_prefix="index"):
    """Saves the TF-IDF components to disk."""
    if vectorizer is None or tfidf_matrix is None or publications is None:
        print("No index to save.")
        return
        
    with open(f"{filename_prefix}_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    with open(f"{filename_prefix}_matrix.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)

    # Save publications list 
    with open(f"{filename_prefix}_publications.pkl", "wb") as f:
        pickle.dump(publications, f)

    print("TF-IDF vectorizer, matrix, and publications saved.")

def main():
    """Main function to run the indexer."""
    publications = load_data("coventry_publications.csv")
    if publications is None:
        return
    
    print("Building TF-IDF index...")
    vectorizer, tfidf_matrix, publications_list = build_tfidf_index(publications)
    
    # condition to handle sparse matrix
    if vectorizer and tfidf_matrix.shape[0] > 0:
        save_index(vectorizer, tfidf_matrix, publications_list)
        print("Indexing complete.")
    else:
        print("Failed to build index.")

if __name__ == "__main__":
    main()