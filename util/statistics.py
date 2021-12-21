import matplotlib.pyplot as plt

def confusion_matrix(labels, predictions, PositiveClassFunction = lambda y: y > 0.5):
	print("\nConfusion Matrix:")
	predictions_positive = PositiveClassFunction(predictions)
	test_positive = PositiveClassFunction(labels)
	true_positive = 0
	false_positive = 0
	true_negative = 0
	false_negative = 0
	total = len(labels)
	for i in range(total):
		if (predictions_positive[i] == test_positive[i]):
			if predictions_positive[i]:
				true_positive += 1
			else:
				true_negative += 1
		else:
			if predictions_positive[i]:
				false_positive += 1
			else:
				false_negative += 1		
	print("True Positive: ", true_positive, "; False Positive: ", false_positive)
	print("False Negative: ", false_negative, "; True Negative: ", true_negative)
	print("Total samples: ", total)
	print("Accuracy: ", (true_positive + true_negative) / total)

	
	precision = "undefined" if true_positive + false_positive == 0 else \
		true_positive / (true_positive + false_positive)
	recall    = "undefined" if true_positive + false_negative == 0 else \
		true_positive / (true_positive + false_negative)
	f1_score  = "undefined" if "undefined" in [precision, recall] or precision+recall == 0 else \
		2*(precision*recall) / (precision+recall)
	print("Precision: ", precision)
	print("Recall: ", recall)
	print("F1 Score: ", f1_score)

def plot(labels, predictions, title, filename):
	fontSize = 14
	fig = plt.figure(num=111, dpi=250)
	ax = fig.add_subplot()
	ax.set_title(title, fontsize=fontSize, pad=10)
	sc = plt.scatter(labels, predictions, c='red' , s=30, marker='o', edgecolors='k', linewidths=0.5, label='Prediction')
	plt.legend(loc='upper left')
	plt.xlabel('True Point Differential', fontsize=15)
	plt.ylabel('Predicted Point Differential', fontsize=15)
	plt.savefig(filename)
	plt.close(fig)