import pandas
import numpy as np

def readTable(filename: str):
	table = pandas.read_csv(filename, index_col=0)
	return table

def balanceDataset(df):
	# NOTE: the data could be mirror here so that there are equal wins and losses for home vs visitor wins;
	# however, this was avoided in case the home-field advantage is a real thing

	indices0 = np.where(df['points0'] > df['points1'])[0]
	indices1 = np.where(df['points0'] < df['points1'])[0]
	len0 = len(indices0)
	len1 = len(indices1)

	np.random.seed(42)
	if len0 > len1:
		remove_amount = len0 - len1
		indices_to_remove = np.random.choice(indices0, size=remove_amount, replace=False)
	elif len0 < len1:
		remove_amount = len1 - len0
		indices_to_remove = np.random.choice(indices1, size=remove_amount, replace=False)
	else:
		return df

	return df.drop(indices_to_remove)

def train_test_split(x, y, train_size):
	np.random.seed(42)
	indices = np.random.permutation(x.shape[0])
	train_len = round(x.shape[0] * train_size)
	training_ind, test_ind = indices[:train_len], indices[train_len:]
	return x.iloc[training_ind], x.iloc[test_ind],  y.iloc[training_ind], y.iloc[test_ind]
