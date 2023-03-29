# training support vector machine model

# making models with different parameters
import numpy as np
import pandas as pd
from sklearn import svm
import pickle
from data_processing import get_data_from_pgn

X1, y1 = get_data_from_pgn('data/lichess_elite.pgn', feature_version=3, num_data=50000)

# X2, y2 = get_data_from_pgn('data/lichess_elite.pgn', feature_version=2, num_data=50000)

model1 = svm.SVR(kernel='rbf', C=1, gamma='auto')

model1.fit(X1, y1)

pickle.dump(model1, open('models/model1.sav', 'wb'))

