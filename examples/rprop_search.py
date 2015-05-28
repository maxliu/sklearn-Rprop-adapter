"""

search for parameters for rpropClassfier

"""


from sklearn.grid_search import RandomizedSearchCV
import numpy as np
import pandas as pd
from scipy.stats import randint as sp_randint
from rpropClassfier import RPClassifier

if __name__ == '__main__':

    h_size = 8
    epo = 3

    print "reading data"
    train = pd.read_csv('../data/train.csv')

    x_train = train.values[:, 0:-1]
    y_train = train.values[:, -1]

    y_train_01 = np.array([1 if yn > 0.5 else 0 for yn in y_train])

    bpc = RPClassifier(h_size=h_size, epo=epo)

    bpc.fit(x_train, y_train_01)

    p = bpc.predict(x_train)

    score = bpc.score(x_train, y_train_01)
    print "score = ", score

    # test gridsearch
    print "starting grid search ... "
    param_dist = {"h_size": sp_randint(2, 10)}
    clf = bpc
    n_iter_search = 2
    random_search = RandomizedSearchCV(clf,
                                       param_distributions=param_dist,
                                       n_iter=n_iter_search,
                                       verbose=1
                                       )

    print("start fitting ....")
    random_search.fit(x_train, y_train_01)
    print("Best parameters set found on development set:")
    print()
    print(random_search.best_estimator_)
    print()
    print("scores on development set:")
    print()
    for params, mean_score, scores in random_search.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
