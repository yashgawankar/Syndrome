import os
import pandas as pd

df = pd.DataFrame()
path = '../forecast/'
for file in os.listdir(path):
    df_ = pd.read_csv(path + file)
    df = pd.concat([df,df_])

df.to_csv('../data/forecast_2020.csv',index=False)
# print(df.shape)
'''
# univariate lstm example
from numpy import array
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# df = pd.read_csv('test_data.csv')

# df = df[(df['Supplier Name']=='Globant') & (df['Function']=='Information Technology And Services') &
#         (df['Service']=='Email Services') & (df['Country']=='New Zealand')]# & (df['Region']=='Christchurch, Canterbury, New Zealand')]
# print(df[['Resources','Year']])
# df.to_csv('test_data1.csv',index=False)




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

pred_df = pd.DataFrame()
df1 = df.drop_duplicates(subset=['Supplier Name', 'Function', 'Region', 'Country', 'Service'])
df1.info()
for i,r in df1.iterrows():
	print(i,' out of ',df1.shape[0])
	df_ = df[(df['Supplier Name']==r['Supplier Name']) & (df['Function']==r['Function']) &
		(df['Service']==r['Service']) & (df['Country']==r['Country']) & (df['Region']==r['Region'])]
	# print(df_)
	pred_list = ['Resources', 'Avg. Cost($)', 'Rating', 'Average Delivery Time', 'Number of Escalations']
	pred_dict = {}
	for pred in pred_list:
		raw_seq = list(df_[pred].values)
		# choose a number of time steps
		n_steps = 3
		batch_size = 10
		# split into samples
		X, y = split_sequence(raw_seq, n_steps)
		# print(y)
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
		# callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5)
		model.fit(X, y, epochs=50, verbose=0)#callbacks=[callback])
		# demonstrate prediction
		x_input = array(X[-1])
		x_input = x_input.reshape((1, n_steps, n_features))
		yhat = model.predict(x_input, verbose=0)
		# print(yhat[0])
		pred_dict[pred] = yhat[0][0]
		pred_dict['Year'] = 2020
	for i_ in ['Supplier Name', 'Function', 'Region', 'Country', 'Service']:
		pred_dict[i_] = r[i_]
	temp_df = pd.DataFrame(pred_dict, index=[0])
	pred_df = pd.concat([pred_df,temp_df])
	print(pred_df)

pred_df = pred_df[['Supplier Name', 'Function', 'Region', 'Country', 'Resources',
       'Service', 'Year', 'Avg. Cost($)', 'Rating', 'Average Delivery Time',
       'Number of Escalations']]
pred_df.to_csv('../data/forecast_2020.csv',index=False)'''