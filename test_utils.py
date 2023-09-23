import pytest
from utils import split_train_dev_test,read_digits

def inc(x):
    return x + 1



def test_inc():
    assert inc(4) == 5

def test_hparam_count():
     gama_ranges = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
     C_ranges = [0.1,1,2,5,10]
     list_of_all_param_combination = [{'gamma': gamma, 'C': C} for gamma in gama_ranges for C in C_ranges]
     assert len(list_of_all_param_combination) == 35

def test_data_splitting():
    X,y = read_digits()
    X = X[:100,:,:]
    y = y[:100]

    test_size = 0.1
    dev_size = 0.6
    train_size = 1 - (dev_size + test_size)

    X_train, X_test,X_dev, y_train, y_test,y_dev = split_train_dev_test(X, y, test_size=test_size, dev_size=dev_size);
    assert len(X_train) == int(train_size * len(X)) and len(X_test) == int(test_size * len(X)) and len(X_dev) == int(dev_size * len(X))
