import os
from util.average_comparisons.dataset import *
from util.dataset import *
from util.average_comparisons.model import *
from util.statistics import confusion_matrix


cache_dir = os.path.join("cache", os.path.basename(__file__).split(".")[0])
if not os.path.exists(cache_dir):
	os.makedirs(cache_dir)

csv_filename = "dataset.csv"
csv_filename = os.path.join(cache_dir, csv_filename)
if not os.path.exists(csv_filename):
	constructDataset(csv_filename)

df = readTable(csv_filename)

df = balanceDataset(df)
x = df[df.columns.drop(['season', 'points0', 'points1'])]
x = standardizeDataset(x)
print(x)
y = (df['points0'] < df['points1']).astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, 0.6)

model = create_model()

saved_weights_file = 'saved_weights'
saved_weights_file = os.path.join(cache_dir, saved_weights_file)
model.load_weights(filepath=saved_weights_file)
# history = train_model(x_train, y_train, model, 3000, 10000)
# model.save_weights(filepath=saved_weights_file)

test_loss, test_acc = model.evaluate(x=x_test, y=y_test)
print("Test_loss: ", test_loss, "; Test_acc: ", test_acc)

predictions = model.predict(x=x_test)

predictions = predictions.flatten()
y_test = y_test.to_numpy()
confusion_matrix(y_test, predictions, lambda y: y > 0.5)
