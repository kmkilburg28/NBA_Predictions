import os
import tensorflow as tf
from util.player_comparisons.dataset import *
from util.dataset import *
from util.player_comparisons.model import *
from util.statistics import confusion_matrix


columns = [
	'fg_per_g', 'fga_per_g', 'fg_pct',
	'fg3_per_g', 'fg3a_per_g', 'fg3_pct',
	'fg2_per_g', 'fg2a_per_g', 'fg2_pct',
	'efg_pct',
	'ft_per_g', 'fta_per_g', 'ft_pct',
	'orb_per_g', 'drb_per_g', 'trb_per_g',
	'ast_per_g', 'stl_per_g', 'blk_per_g', 'tov_per_g', 'pf_per_g', 'pts_per_g'
]

cache_dir = os.path.join("cache", os.path.basename(__file__).split(".")[0])
if not os.path.exists(cache_dir):
	os.makedirs(cache_dir)

csv_filename = "dataset.csv"
csv_filename = os.path.join(cache_dir, csv_filename)
if not os.path.exists(csv_filename):
	constructDataset(csv_filename, columns)

df = readTable(csv_filename)

df = fixNumpyStrings(df)
df = balanceDataset(df)
df = standardizeDataset(df, cache_dir, columns)
x = df[df.columns.drop(['season', 'points0', 'points1'])]
print(x)
y = (df['points0'] < df['points1']).astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, 0.6)


input0 = tf.ragged.constant([nparray.tolist() for nparray in x_train['team_stats0']]).to_tensor()
input1 = tf.ragged.constant([nparray.tolist() for nparray in x_train['team_stats1']]).to_tensor()

model = create_model(int(input0.shape[1]), int(input0.shape[2]))

saved_weights_file = 'saved_weights'
saved_weights_file = os.path.join(cache_dir, saved_weights_file)
model.load_weights(filepath=saved_weights_file)
# history = train_model({'team0': input0, 'team1': input1}, y_train, model, 100, 1000)
# model.save_weights(filepath=saved_weights_file)

input0 = tf.ragged.constant([nparray.tolist() for nparray in x_test['team_stats0']]).to_tensor()
input1 = tf.ragged.constant([nparray.tolist() for nparray in x_test['team_stats1']]).to_tensor()
test_loss, test_acc = model.evaluate(x={'team0': input0, 'team1': input1}, y=y_test)
print("Test_loss: ", test_loss, "; Test_acc: ", test_acc)

predictions = model.predict(x={'team0': input0, 'team1': input1})

predictions = predictions.flatten()
y_test = y_test.to_numpy()
confusion_matrix(y_test, predictions, lambda y: y > 0.5)
