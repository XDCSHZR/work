from sklearn.base import BaseEstimator, clone

# M_y(x) regression wrapper
class RegressionWrapper(BaseEstimator):
    def __init__(self, clf):
        self.clf = clf

    def fit(self, X, y, **kwargs):
        self.clf_ = clone(self.clf)
        self.clf_.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.clf_.predict_proba(X)[:, 1]