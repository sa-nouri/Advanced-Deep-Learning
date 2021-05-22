import pandas as pd
from coclust.evaluation.external import accuracy
from tensorflow.keras import models

test_labels = pd.read_csv('test_labels.csv')
data = pd.read_csv('data.csv')

model = models.load_model('/model_name/')

predicted_labels = model.predict()
hungarian_accuracy = accuracy(test_labels, predicted_labels)

print(f'The Hungarian Accuracy is {hungarian_accuracy}')
