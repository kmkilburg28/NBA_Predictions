import tensorflow as tf

def create_model():
	model = tf.keras.models.Sequential([
		tf.keras.layers.Dense(units=128, activation='relu'),
		tf.keras.layers.Dense(units=64, activation='relu'),
		tf.keras.layers.Dense(units=32, activation='relu'),
		tf.keras.layers.Dense(units=1, activation='sigmoid')
	])

	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
		loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
		metrics=['binary_accuracy']
	)
	return model

def train_model(train_features, train_labels, model, epochs=30, batch_size=100, validation_split=0.1):

	history = model.fit(x=train_features, y=train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
	return history