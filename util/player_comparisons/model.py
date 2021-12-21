import tensorflow as tf

def create_model(max_length: int, width: int):
	print(max_length)
	print(width)
	team1_input = tf.keras.Input(shape=(max_length, width,), name='team0')
	team2_input = tf.keras.Input(shape=(max_length, width,), name='team1')

	rnn = tf.keras.layers.GRU(128)
	team1_rnn = rnn(team1_input)
	team2_rnn = rnn(team2_input)

	concat_layer = tf.keras.layers.Concatenate()([team1_rnn, team2_rnn])
	dense_layer = tf.keras.layers.Dense(units=128, activation='relu')(concat_layer)
	dense_layer = tf.keras.layers.Dense(units=64, activation='relu')(dense_layer)
	dense_layer = tf.keras.layers.Dense(units=32, activation='relu')(dense_layer)
	pred_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')(dense_layer)

	model = tf.keras.Model(
		inputs=[team1_input, team2_input],
		outputs=[pred_layer]
	)


	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
		loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
		metrics=['binary_accuracy']
	)
	return model

def train_model(train_features, train_labels, model, epochs=30, batch_size=100, validation_split=0.1):

	history = model.fit(x=train_features, y=train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
	return history
