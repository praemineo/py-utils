"""
Helper functions for using while making a neural network using tensorflow
"""

import tensorflow as tf
import sys
import numpy as np
import time
import os
import re


class Model():
    def __init__(self):
        pass
        # self.graph.as_default()

    def init(self):
        self.graph = tf.Graph()
        return self.graph

    def graph_info(self):
        return self.graph.get_operations()

    def session(self):
        self.sess = tf.Session()
        return self.sess

    def train(self,ops,x,y,x_data,y_data,num_epochs=1,batch_size = 1):

        len_data = len(x_data)


        with self.graph.as_default():
            for epoch in range(1,num_epochs+1):
                epoch_start_time = time.time()
                for batch_start in range(0,len_data,batch_size):
                    optimiser_value,loss_value = self.sess.run(ops,feed_dict={x:self.next_batch(x_data,batch_start,batch_size),y:self.next_batch(y_data,batch_start,batch_size)})
                    sys.stdout.write("\rEpoch: {}/{}, Batch: {}/{}, Training loss: {}".format(epoch,num_epochs,(batch_start/batch_size)+1,len_data/batch_size,np.mean(loss_value)))
                    sys.stdout.flush()
                print " Time: {} s".format(time.time()-epoch_start_time)

    def get_latest_checkpoint(self,checkpoint_path="./model_weights/"):
        checkpoints = os.listdir(checkpoint_path)
        latest_checkpoint_index = -1
        for checkpoint in checkpoints:
            if "meta" in checkpoint:
                checkpoint_index = int(re.findall("\d+",checkpoint)[0])
                if checkpoint_index > latest_checkpoint_index:
                    latest_checkpoint_index = checkpoint_index
        return latest_checkpoint_index

    def restore_weights(self,checkpoint_path="./model_weights/"):
        latest_checkpoint = self.get_latest_checkpoint(checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(self.sess,checkpoint_path+"model_weights_{}.ckpt".format(latest_checkpoint))




    def next_batch(self, data, batch_start, batch_size):
        len_data = len(data)
        batch_end = min(batch_start+batch_size,len_data)
        return data[batch_start:batch_end]

    def save_weights(self,checkpoint_path = None,checkpoint_number=None):
        if checkpoint_path is None:
            print "Checkpoint path not provided. Saving to ./model_weights/"
            checkpoint_path = "./model_weights/"
        if not os.path.exists(checkpoint_path):
            print "Checkpoint path not found. Making directory.."
            os.makedirs(checkpoint_path)

        if checkpoint_number is None:
            checkpoint_number = self.get_latest_checkpoint(checkpoint_path)+1

        saver = tf.train.Saver()
        save_path = saver.save(self.sess,checkpoint_path+"model_weights_{}.ckpt".format(checkpoint_number))
        print "Model saved at",save_path
