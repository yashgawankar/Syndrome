import os
import pandas as pd

# df = pd.DataFrame()
# path = '../data/'
# for file in os.listdir(path):
#     df_ = pd.read_csv(path + file)
#     df = pd.concat([df,df_])

# print(df.shape)

# univariate lstm example
from numpy import array
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

df = pd.read_csv('test_data.csv')

# df = df[(df['Supplier Name']=='Globant') & (df['Function']=='Information Technology And Services') &
#         (df['Service']=='Email Services') & (df['Country']=='New Zealand') & (df['Region']=='Christchurch, Canterbury, New Zealand')]
# print(df[['Resources','Year']])
# df.to_csv('test_data.csv',index=False)



# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
# raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
raw_seq = list(df['Resources'].values)
# choose a number of time steps
n_steps = 3
batch_size = 10
# split into samples
X, y = split_sequence(raw_seq, n_steps)
print(y)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
# model.add(Dense(96))
# model.add(Dense(1, activation='relu'))
optimizer = keras.optimizers.Adam(lr=.1)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
# model.compile(optimizer='adam', loss='mse')
# fit model
callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5)
model.fit(X, y, epochs=100, verbose=1)#callbacks=[callback])
# demonstrate prediction
x_input = array(X[-1])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=1)
print(yhat[0])