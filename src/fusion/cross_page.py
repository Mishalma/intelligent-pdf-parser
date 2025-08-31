from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def detect_headers_footers(all_pages_elements, sim_threshold=0.8, var_threshold=20):
    # Collect candidate bands (top/bottom text elements across pages)
    candidates = []
    for page_num, elements in enumerate(all_pages_elements):
        for el in elements:
            if el['type'] == 'text':
                y_mid = (el['bbox'][1] + el['bbox'][3]) / 2
                candidates.append({'page': page_num, 'y_mid': y_mid, 'text': el['text'], 'bbox': el['bbox']})
    
    # Cluster by y-position variance
    y_positions = np.array([c['y_mid'] for c in candidates]).reshape(-1, 1)
    clusters = DBSCAN(eps=var_threshold, min_samples=2).fit_predict(y_positions)
    
    # Filter clusters with high text similarity
    headers_footers = []
    vectorizer = TfidfVectorizer()
    for cluster_id in set(clusters):
        if cluster_id == -1: continue
        cluster_cands = [c for i, c in enumerate(candidates) if clusters[i] == cluster_id]
        texts = [c['text'] for c in cluster_cands]
        tfidf = vectorizer.fit_transform(texts)
        sim_matrix = (tfidf * tfidf.T).toarray()
        if np.mean(sim_matrix) > sim_threshold:
            label = 'header' if np.mean([c['y_mid'] for c in cluster_cands]) < 100 else 'footer'  # Adjust threshold based on page height
            for c in cluster_cands:
                headers_footers.append({'label': label, 'bbox': c['bbox'], 'page': c['page']})
    
    return headers_footers