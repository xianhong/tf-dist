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


def prepare(TASK_ID,BATCH_SIZE,NUM_WORKERS):
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
	SEED = tf.constant(23,dtype=tf.int64)
	train_datasets_unbatched = train_dataset.repeat().shuffle(BUFFER_SIZE,seed = SEED)

	options = tf.data.Options()
	options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
	train_datasets_unbatched = train_datasets_unbatched.with_options(options)

	train_datasets = train_datasets_unbatched.batch(BATCH_SIZE,drop_remainder=True)
	return train_datasets





if __name__=='__main__':
	parser=argparse.ArgumentParser(description="Worker")
	parser.add_argument('--cont',type=int,default=0,help='Whether to continue training :0(default) for training from start; 1 for continue training')
	parser.add_argument('--batch',type=int,default=128,help='Batch size per worker')
	parser.add_argument('--task',type=int,required=True,help='Worker task ID')
	parser.add_argument('--workers',type=int,default=2,metavar='N',help='# of Workers')
	parser.add_argument('--epochs',type=int,default=6,help='Epochs')
	args=parser.parse_args()
	os.environ["TF_CONFIG"] = json.dumps({
	    "cluster": {
		"worker": ["worker0:7777", "worker1:7777"],
	    },
	   "task": {"type": "worker", "index": args.task}
	})
	strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
	    tf.distribute.experimental.CollectiveCommunication.AUTO)

	GLOBAL_BATCH_SIZE = args.batch * args.workers
	train_datasets = prepare(TASK_ID=args.task,BATCH_SIZE = GLOBAL_BATCH_SIZE,NUM_WORKERS = args.workers)

	callbacks = [keras.callbacks.ModelCheckpoint(filepath='./keras-ckpt',save_weights_only=True)]
	with strategy.scope():
		multi_worker_model = build_and_compile_autoencoder()
		if (args.cont != 0):
			multi_worker_model.load_weights('./keras-ckpt')
	multi_worker_model.fit(x=train_datasets, steps_per_epoch = 60000 //GLOBAL_BATCH_SIZE ,epochs=args.epochs, callbacks=callbacks)
		


