import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def compute_domain_separability(source_features, target_features, seed=6304):
    n_samples = min(len(source_features), len(target_features))
    np.random.seed(seed)
    idx_s = np.random.choice(len(source_features), n_samples, replace=False)
    idx_t = np.random.choice(len(target_features), n_samples, replace=False)
    X_s = source_features[idx_s]
    X_t = target_features[idx_t]
    X = np.vstack([X_s, X_t])
    y = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    separability_score = accuracy_score(y_test, y_pred) * 100.0
    return separability_score
