import tensorflow as tf

def create_model():
	model = tf.keras.models.Sequential([
		tf.keras.layers.Dense(units=128, activation='relu'),
		tf.keras.layers.Dense(units=64, activation='relu'),
		tf.keras.layers.Dense(units=32, activation='relu'),
		tf.keras.layers.Dense(units=1)
	])

	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
		loss=tf.keras.losses.MeanSquaredError(),
		metrics=['accuracy']
	)
	return model
