from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from scipy.special import expit
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier


def run_multiclass_naive_bayes(X_train, y_train, X_test):
    # Each transcript can have one label.

    clf = MultinomialNB(alpha=0.001, fit_prior=False)
    clf.fit(X_train, np.argmax(y_train, 1))

    y_pred_probs = clf.predict_proba(X_test)

    nmissing_classes = y_train.shape[1] - y_pred_probs.shape[1]
    if nmissing_classes:
        y_missing = np.zeros((y_pred_probs.shape[0], nmissing_classes))
        y_pred_probs = np.concatenate((y_pred_probs, y_missing), axis=1)

    y_pred_mat = y_pred_probs > 0.1

    return y_pred_mat, y_pred_probs


def run_binary_naive_bayes(X_train, y_train, X_test):
    # Consider applying each label to every transcript, i.e., a binary decision per code

    nclasses = y_train.shape[1]

    y_pred_mat = np.zeros((X_test.shape[0], nclasses))
    y_pred_probs = np.zeros((X_test.shape[0], nclasses))

    for c in range(nclasses):
        # iterate over the classes and make a classifier for each class

        clf = MultinomialNB(alpha=0.001)
        clf.fit(X_train, y_train[:, c])

        y_te_pred_c = clf.predict(X_test)
        # y_pred_mat[:, c] = y_te_pred_c

        y_pred_probs[:, c] = clf.predict_proba(X_test)[:, 1]
        y_pred_mat[:, c] = y_pred_probs[:, c] > 0.1

    return y_pred_mat, y_pred_probs


def run_multiclass_svm(X_train, y_train, X_test):
    nclasses = y_train.shape[1]

    clf = SVC(kernel='rbf', C=2)

    clf.fit(X_train, np.argmax(y_train, 1))

    y_pred = clf.predict(X_test)
    y_pred_probs = clf.decision_function(X_test)

    nmissing_classes = y_train.shape[1] - y_pred_probs.shape[1]
    if nmissing_classes:
        y_missing = np.zeros((y_pred_probs.shape[0], nmissing_classes))
        y_pred_probs = np.concatenate((y_pred_probs, y_missing), axis=1)

    y_pred_mat = np.zeros((y_pred.shape[0], nclasses))
    y_pred_mat[np.arange(y_pred.shape[0]), y_pred] = 1

    return y_pred_mat, y_pred_probs


def run_binary_svm(X_train, y_train, X_test):
    nclasses = y_train.shape[1]

    y_pred_mat = np.zeros((X_test.shape[0], nclasses))
    y_pred_probs = np.zeros((X_test.shape[0], nclasses))

    for c in range(nclasses):
        # iterate over the classes and make a classifier for each class

        clf = SVC(kernel='rbf', C=2)
        clf.fit(X_train, y_train[:, c])

        y_te_pred_c = clf.predict(X_test)

        y_pred_probs[:, c] = clf.decision_function(X_test)

        y_pred_mat[:, c] = expit(y_pred_probs[:, c]) > 0.1

    return y_pred_mat, y_pred_probs


def run_nearest_centroid(X_train, y_train, X_test):
    nc_clf = NearestCentroid(metric='euclidean')
    nc_clf.fit(X_train, np.argmax(y_train, 1))
    y_pred = nc_clf.predict(X_test)

    y_pred_mat = np.zeros((y_pred.shape[0], y_train.shape[1]))
    y_pred_mat[np.arange(y_pred.shape[0]), y_pred] = 1

    return y_pred_mat, y_pred_mat


def run_nearest_neighbors(X_train, y_train, X_test, n_neighbors=3):
    kn_clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', metric='cosine')
    kn_clf.fit(X_train, np.argmax(y_train, 1))

    y_pred = kn_clf.predict(X_test)
    y_pred_mat = np.zeros((y_pred.shape[0], y_train.shape[1]))
    y_pred_mat[np.arange(y_pred.shape[0]), y_pred] = 1

    y_probs = kn_clf.predict_proba(X_test)

    nmissing_classes = y_desc.shape[1] - y_probs.shape[1]
    if nmissing_classes:
        y_missing = np.zeros((y_probs.shape[0], nmissing_classes))
        y_probs = np.concatenate((y_probs, y_missing), axis=1)

    return y_pred_mat, y_probs