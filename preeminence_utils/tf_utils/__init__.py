"""
Helper functions for using while making a neural network using tensorflow
"""

import tensorflow as tf
import sys
import numpy as np
import time
import os
import re


class Model:
    def __init__(self):
        self.graph = None
        self.sess = None
        self.writer = None

    def init(self):
        """
        Initialise a new model and return its graph.
        This function will spawn a new graph and return it.
        You'll have to set it to default graph in order to
        add ops to it.
        Sample:
        model = tf_utils.Model()
        model_graph = model.init().as_default()

        :return: Returns a new graph.
        """
        self.graph = tf.Graph()
        return self.graph

    def graph_info(self):
        """
        Get ops in the graph

        :return: Graph ops
        """
        return self.graph.get_operations()

    def visualise(self, logdir="./logs"):
        """
        Save graph summary in the logidr to be
        visualised by tensorboard.
        Summaries for individual ops to be added.

        :param logdir: Destination for storing graph logs. ./logs by default
        :return: Nothing
        """
        self.writer = tf.summary.FileWriter(logdir)
        self.writer.add_graph(self.sess.graph)

    def session(self):
        """
        Create and return a new session for training.

        :return: New session object
        """
        self.sess = tf.Session()
        return self.sess

    def train(self, ops, x, y, x_data, y_data, num_epochs=1, batch_size=1):
        """
        Training function. Executes the graph on a given dataset.

        :param ops: Graph ops to be calculated and returned. Must be [optimiser_op,loss_op]
        :param x: placeholder tensor for x
        :param y: placeholder tensor for y
        :param x_data: Training data to be fed into x
        :param y_data: Training data to be fed into y
        :param num_epochs: Number of epochs to be executed
        :param batch_size: Size of a batch to be fed at a time
        :return: Nothing
        """

        len_data = len(x_data)

        with self.graph.as_default():
            for epoch in range(1, num_epochs + 1):
                epoch_start_time = time.time()
                for batch_start in range(0, len_data, batch_size):
                    optimiser_value, loss_value = self.sess.run(ops, feed_dict={
                        x: self.next_batch(x_data, batch_start, batch_size),
                        y: self.next_batch(y_data, batch_start, batch_size)})
                    sys.stdout.write(
                        "\rEpoch: {}/{}, Batch: {}/{}, Training loss: {}".format(
                            epoch,
                            num_epochs,
                            (batch_start / batch_size) + 1,
                            len_data / batch_size,
                            np.mean(loss_value)
                        )
                    )
                    sys.stdout.flush()
                print " Time: {} s".format(time.time() - epoch_start_time)

    def get_latest_checkpoint(self, checkpoint_path="./model_weights/"):
        """
        Get the name of the latest checkpoint in the checkpoints
        directory in order to load the latest weights to continue
        training for that point.

        :param checkpoint_path: Custom directory where checkpoints are saved
        :return:
        """
        checkpoints = os.listdir(checkpoint_path)
        latest_checkpoint_index = -1
        for checkpoint in checkpoints:
            if "meta" in checkpoint:
                checkpoint_index = int(re.findall("\d+", checkpoint)[0])
                if checkpoint_index > latest_checkpoint_index:
                    latest_checkpoint_index = checkpoint_index
        return latest_checkpoint_index

    def restore_weights(self, checkpoint_path="./model_weights/"):
        """
        Restore weights from the checkpoint path to the latest
        checkpoint to resume training from that point.

        :param checkpoint_path: Custom directory where checkpoints are saved
        :return:
        """
        if not os.path.exists(checkpoint_path):
            raise OSError("Checkpoint directory not found")
        latest_checkpoint = self.get_latest_checkpoint(checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(self.sess, checkpoint_path + "model_weights_{}.ckpt".format(latest_checkpoint))

    def next_batch(self, data, batch_start, batch_size):
        """
        Get next batch from the training data
        This should be a generator function :/

        :param data: Training data
        :param batch_start: batch start index
        :param batch_size: size of the batch
        :return:
        """
        len_data = len(data)
        batch_end = min(batch_start + batch_size, len_data)
        return data[batch_start:batch_end]

    def save_weights(self, checkpoint_path=None, checkpoint_number=None):
        """
        Save the current weights of the model to disk at the checkpoint
        path.
        :param checkpoint_path: Custom directory where checkpoints are saved
        :param checkpoint_number: Custom number to append at the end of checkpoint
        :return:
        """
        if checkpoint_path is None:
            print "Checkpoint path not provided. Saving to ./model_weights/"
            checkpoint_path = "./model_weights/"
        if not os.path.exists(checkpoint_path):
            print "Checkpoint path not found. Making directory.."
            os.makedirs(checkpoint_path)

        if checkpoint_number is None:
            checkpoint_number = self.get_latest_checkpoint(checkpoint_path) + 1

        saver = tf.train.Saver()
        save_path = saver.save(self.sess, checkpoint_path + "model_weights_{}.ckpt".format(checkpoint_number))
        print "Model saved at", save_path
