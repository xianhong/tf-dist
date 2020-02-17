### Demonstrate different Tensorflow (2.1) training topologies for training a Keras model.
Training topologies inclde:
#### Single worker ( CPUs plus an optional single GPU)
  Change directory to `cpu_or_single_gpu`, and run:   
  `docker run -it --rm  -v $PWD:/notebooks yangxh/lab:tf-2.1 python AE_worker.py --epochs EPOCHS --batch BATCH`
#### Single worker with Cloud TPU (in Colab environment)
#### Multi-worker distributed training 
 - (inefficient) Data sharding at the end of input pipeline inside Keras fit method ('auto_shard_policy' set to 'DATA')  
   Change directory to `multi-worker-DATA-SHARD`  
 - Manual data sharding at early stage of input pipeline outside Keras fit method ('auto_shard_policy' set to 'OFF')  
    Change directory to `multi-worker-MANUAL-SHARD`  
   <br/>
   On the 1st. terminal run:  
   `docker run -it --rm --network my-net --name worker0 -v $PWD:/notebooks yangxh/lab:tf-2.1 python AE_worker.py --task=0 --batch BATCH --epochs EPOCHS`
   <br/>
   And, on 2nd. terminal run:  
   `docker run -it --rm --network my-net --name worker1 -v $PWD:/notebooks yangxh/lab:tf-2.1 python AE_worker.py --task=1 --batch BATCH --epochs EPOCHS`

