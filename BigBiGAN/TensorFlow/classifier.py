import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

base_path: str = 'path-name'

train_data = pd.read_csv(base_path+'/train.csv', header=None)
test_data = pd.read_csv(base_path+'/test.csv', header=None)

train_label = np.array(train_data.pop('label'))
test_label = np.array(test_data.pop('label'))

clf = make_pipeline(StandardScaler(),
                    LinearSVC(random_state=0, tol=1e-5))
clf.fit(np.array(train_data), train_label)

predictions = clf.predict(np.array(test_data))
print(f'The Linear Accuracy is {accuracy_score(np.array(test_label), predictions)}')

