import os
import json
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import argparse
def build_and_compile_autoencoder():
  	model = tf.keras.Sequential([layers.Conv2D(16, 3, activation='relu',input_shape=(28, 28, 1)),
	                             layers.Conv2D(32, 3, activation='relu'),
	                             layers.MaxPooling2D(3),
	                             layers.Conv2D(32, 3, activation='relu'),
	                             layers.Conv2D(16, 3, activation='relu'),
	                             layers.GlobalMaxPooling2D(),
	                             layers.Reshape((4, 4, 1)),
	                             layers.Conv2DTranspose(16, 3, activation='relu'),
	                             layers.Conv2DTranspose(32, 3, activation='relu'),
	                             layers.UpSampling2D(3),
	                             layers.Conv2DTranspose(16, 3, activation='relu'),
	                             layers.Conv2DTranspose(1, 3, activation='relu')])
  	model.compile(optimizer='adam',
	      loss='mean_squared_error',
	      metrics=['mean_squared_error'])
  	return model


def prepare(BATCH_SIZE):
	mnist = keras.datasets.mnist

	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0
	x_train = x_train.reshape((-1, 28,28,1))
	x_test = x_test.reshape((-1, 28,28,1))
	def gen(x):
	  for i in range(len(x)):
	    yield (x[i],x[i])
	train_dataset = tf.data.Dataset.from_generator(lambda :gen(x_train),(tf.float32,tf.float32), (tf.TensorShape([28,28,1]),tf.TensorShape([28,28,1])))

	BUFFER_SIZE = 2000
	train_datasets_unbatched = train_dataset.repeat().shuffle(BUFFER_SIZE)
	train_datasets = train_datasets_unbatched.batch(BATCH_SIZE,drop_remainder=True)
	return train_datasets

if __name__=='__main__':
	parser=argparse.ArgumentParser(description="Worker")
	parser.add_argument('--cont',type=int,default=0,help='Whether to continue training :0(default) for training from start; 1 for continue training')
	parser.add_argument('--batch',type=int,default=128,help='Batch size')
	parser.add_argument('--epochs',type=int,default=6,help='Epochs')
	args=parser.parse_args()
	 
	train_datasets = prepare(args.batch)

	callbacks = [keras.callbacks.ModelCheckpoint(filepath='./keras-ckpt',save_weights_only=True)]
	model = build_and_compile_autoencoder()
	if (args.cont != 0):
		model.load_weights('./keras-ckpt')
	model.fit(x=train_datasets, steps_per_epoch = 60000 //args.batch ,epochs=args.epochs, callbacks=callbacks)
		


