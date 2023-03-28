# training support vector machine model

# making models with different parameters
import numpy as np
import pandas as pd
from sklearn import svm
import pickle
from data_processing import get_data_from_pgn

# load data from csv file
X1, y1 = get_data_from_pgn('data/lichess_elite.pgn', feature_version=1, num_data=50000)

X2, y2 = get_data_from_pgn('data/lichess_elite.pgn', feature_version=2, num_data=50000)

model1 = svm.SVR(kernel='linear', C=1, gamma='auto')

model1.fit(X1, y1)

pickle.dump(model1, open('models/model1.sav', 'wb'))

model2 = svm.SVR(kernel='linear', C=1, gamma='auto')

model2.fit(X2, y2)

pickle.dump(model2, open('models/model2.sav', 'wb'))
