"""
Helper functions for using while making a neural network using tensorflow
"""

import tensorflow as tf
import sys
import numpy as np
import time
import os
import boto3
import inspect


class Model:
    def __init__(self):
        self.graph = None
        self.sess = None
        self.writer = None
        self.saver = None

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
        return tf.train.latest_checkpoint(checkpoint_path)

    def restore_weights(self, checkpoint_path="./model_weights/", s3=False, s3_path = None):
        """
        Restore weights from the checkpoint path to the latest
        checkpoint to resume training from that point.

        :param s3: Flag to decide to download checkpoints from S3 or not
        :param s3_path: Name of subfolder in preeminence-models bucket
        :param checkpoint_path: Custom directory where checkpoints are saved
        :return:
        """

        if s3:
            if s3_path is None:
                raise OSError("s3_path argument not provided.")
            bucket_name = "preeminence-models"  # replace with your bucket name
            checkpoint_key = s3_path+'/checkpoint'  # replace with your object key

            s3_obj = boto3.client('s3')

            try:
                s3_obj.download_file(bucket_name,checkpoint_key, "{}/{}".format(checkpoint_path,'checkpoint'))
                checkpoint_file = open(checkpoint_path+'checkpoint').read().split()[1].strip('"')
                for weights_file in s3_obj.list_objects(Bucket=bucket_name)["Contents"]:
                    if s3_path+"/"+checkpoint_file in  weights_file["Key"]:
                        weights_file_key = weights_file["Key"]
                        weights_file_filename = weights_file["Key"].replace(s3_path+"/","",1)
                        s3_obj.download_file(bucket_name, weights_file_key, "{}/{}".format(checkpoint_path, weights_file_filename))

                print "Downloaded latest checkpoint {} from S3 and restored to model".format(checkpoint_file)


            except Exception as e:
                print e
                raise OSError("File with key: {} not found in bucket: {}".format(checkpoint_key,bucket_name))


        if not os.path.exists(checkpoint_path):
            raise OSError("Checkpoint directory not found")
        latest_checkpoint = self.get_latest_checkpoint(checkpoint_path)
        # print latest_checkpoint
        if latest_checkpoint:
            saver = tf.train.Saver()
            saver.restore(self.sess, latest_checkpoint)
        else:
            raise OSError("Checkpoint file not found at "+checkpoint_path)

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

    def save_weights(self, checkpoint_path=None, checkpoint_number=None, s3=False, s3_path = None):
        """
        Save the current weights of the model to disk at the checkpoint
        path.
        :param s3: Flag to decide to upload weights to S3 or not
        :param s3_path: Subfolder in S3 bucket
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
            latest_checkpoint = self.get_latest_checkpoint(checkpoint_path=checkpoint_path)
            # print latest_checkpoint
            if latest_checkpoint:
                checkpoint_number = int(latest_checkpoint.split("-")[-1])+1
            else:
                checkpoint_number=1

        if self.saver is None:
            self.saver = tf.train.Saver()

        weight_file_prefix = "model_weights.ckpt"
        save_path = self.saver.save(self.sess, checkpoint_path + weight_file_prefix,global_step=checkpoint_number)
        print "Model saved at", save_path
        if s3:
            start_time = time.time()
            if s3_path is None:
                raise OSError("s3_path argument not provided.")
            else:
                s3_obj = boto3.client('s3')
                response = s3_obj.list_buckets()
                bucket_name = "preeminence-models"

                # Get a list of all bucket names from the response
                buckets = [bucket['Name'] for bucket in response['Buckets']]

                if bucket_name in buckets:
                    filename_prefix = weight_file_prefix+"-"+str(checkpoint_number)
                    weight_files = [filename for filename in os.listdir(checkpoint_path) if filename_prefix in filename]
                    s3_obj.upload_file(checkpoint_path + "checkpoint", bucket_name, "{}/{}".format(s3_path, "checkpoint"))
                    for weight_file in weight_files:
                        # print "Uploading",weight_file
                        s3_obj.upload_file(checkpoint_path+weight_file,bucket_name,"{}/{}".format(s3_path,weight_file))
                    print "Model uploaded to S3 at {}/{}/ in {} seconds".format(bucket_name,s3_path,time.time()-start_time)
                else:
                    raise OSError(
                        "Bucket not found. Check if bucket with name {} is present in your AWS account or change the bucket name.".format(
                            bucket_name))