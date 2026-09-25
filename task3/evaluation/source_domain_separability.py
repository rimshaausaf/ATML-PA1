import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def compute_source_domain_separability(source_val_features_dict, seed=6304):
    np.random.seed(seed)
    min_samples = min(len(feats) for feats in source_val_features_dict.values())
    domains = ['photo', 'art_painting', 'cartoon']
    X_list = []
    y_list = []
    for idx, d in enumerate(domains):
        feats = source_val_features_dict[d]
        sub_idx = np.random.choice(len(feats), min_samples, replace=False)
        X_list.append(feats[sub_idx])
        y_list.append(np.full(min_samples, idx))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    clf = LogisticRegression(C=1.0, multi_class='multinomial', max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    separability_score = accuracy_score(y_test, preds) * 100.0
    return separability_score
