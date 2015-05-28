"""

test cases for rpropClassifier

"""

from sklearn.utils.testing import assert_array_equal

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np
import pandas as pd
from rpropClassfier import RPClassifier


def test_and():
    x_train = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_train = np.array([0, 0, 0, 1])
    bpc = RPClassifier(h_size=2, epo=50)
    bpc.fit(x_train, y_train)
    p_c = bpc.predict(x_train)
    assert_array_equal(y_train, p_c)

def test_or():
    x_train = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_train = np.array([0, 1, 1, 1])
    bpc = RPClassifier(h_size=2, epo=50)
    bpc.fit(x_train, y_train)
    p_c = bpc.predict(x_train)
    assert_array_equal(y_train, p_c)

def test_xor():
    x_train = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_train = np.array([1, 0, 0, 0])
    bpc = RPClassifier(h_size=2, epo=50)
    bpc.fit(x_train, y_train)
    p_c = bpc.predict(x_train)
    assert_array_equal(y_train, p_c)

if __name__ == '__main__':
    import pytest
    pytest.main("-v")
