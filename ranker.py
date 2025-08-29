import numpy as np
import pickle
import os

def load_index(filename_prefix="index"):
    """Loads the TF-IDF components from disk."""
    try:
        with open(f"{filename_prefix}_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        
        with open(f"{filename_prefix}_matrix.pkl", "rb") as f:
            tfidf_matrix = pickle.load(f)

        with open(f"{filename_prefix}_publications.pkl", "rb") as f:
            publications = pickle.load(f)
            
        print("TF-IDF vectorizer, matrix, and publications loaded.")
        return vectorizer, tfidf_matrix, publications
    except FileNotFoundError:
        print(f"Error: One or more index files were not found. Please run indexer.py first.")
        return None, None, None

def search_publications_tfidf(query, vectorizer, tfidf_matrix, publications):
    """
    Searches publications using TF-IDF and cosine similarity,
    but boosts results where the query appears in the title.
    """
    query_vec = vectorizer.transform([query])
    cosine_similarities = (tfidf_matrix @ query_vec.T).toarray().ravel()
    
    results = []
    query_lower = query.lower()

    for idx, pub in enumerate(publications):
        score = cosine_similarities[idx]
        if score > 0:
            title = pub['title'].lower()

            if query_lower in title:
                score += 2.0  

            elif any(word in title for word in query_lower.split()):
                score += 1.0  

            pub_copy = pub.copy()
            pub_copy["score"] = round(float(score), 4)
            results.append(pub_copy)
    
    # Sort by boosted score
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results

def main():
    """Main function to run the search engine."""
    vectorizer, tfidf_matrix, publications = load_index()
    if vectorizer is None or tfidf_matrix is None or publications is None:
        return
    
    print("\n--- TF-IDF Search Engine (type 'exit' to quit) ---")
    while True:
        user_query = input("Enter your search query: ")
        if user_query.lower() == 'exit':
            break

        results = search_publications_tfidf(user_query, vectorizer, tfidf_matrix, publications)

        if results:
            print(f"\nFound {len(results)} results for '{user_query}':")
            for i, pub in enumerate(results):
                authors_names = [a['name'] for a in pub['authors']]
                print(f"{i+1}. Title: {pub['title']}")
                print(f"   Authors: {', '.join(authors_names)}")
                print(f"   Date: {pub['date']}")
                print(f"   Score: {pub['score']}")
                print(f"   Abstract: {pub['abstract']}")
                print(f"   Link: {pub['publication_link']}")
                print("-" * 30)
        else:
            print("No matching publications found.")

if __name__ == "__main__":
    main()
