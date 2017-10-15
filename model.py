import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib




from dataClean import prepareData, preproc

freq1, graph1 = prepareData('FrequencyEvents.txt', 620, 715)
freq2, graph2 = prepareData('TruthFrequencyEvents.txt', 620, 715)
graph3 = preproc(graph1, graph2, freq1, freq2)

X = graph3.as_matrix(['amp1', 'freq1'])
Y = graph3.as_matrix(['amp2', 'freq2'])
# print(np.concatenate((X,Y), axis=1))


from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(12, input_dim=2, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=100, batch_size=10)


freq1, graph1 = prepareData('FrequencyEvents.txt', 400, 450)
freq2, graph2 = prepareData('TruthFrequencyEvents.txt', 400, 450)
graph3 = preproc(graph1, graph2, freq1, freq2)

X = graph3.as_matrix(['amp1', 'freq1'])
Y = graph3.as_matrix(['amp2', 'freq2'])

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))