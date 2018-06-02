"""
Helper functions for using while making a neural network using tensorflow
"""

import tensorflow as tf
import sys
import numpy as np
import time
from . import aws_utils
import os
import boto3
from . import utils


class Model:
    def __init__(self,name):
        self.graph = None
        self.sess = None
        self.writer = None
        self.saver = None
        self.name = name

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

    def visualise(self, logdir=None):
        """
        Save graph summary in the logdir to be visualised by tensorboard.
        Merge all summary ops into one. Summaries for individual ops to be added.
        Use self.summary_writer to write summaries to disk. Call this function
        just before starting training. This function adds default graph to the
        summary to be viewed in tensorboard.

        :param logdir: Destination for storing graph logs. ./logs by default
        :return: Nothing
        """

        if logdir is None:
            logdir = "./logs/"+self.name

        self.merged_summary_ops = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(logdir)
        self.summary_writer.add_graph(tf.get_default_graph())

    def session(self):
        """
        Create and return a new session for training.

        :return: New session object
        """
        self.sess = tf.Session()
        return self.sess



    def get_latest_checkpoint(self, checkpoint_path="./model_weights/"):
        """
        Get the name of the latest checkpoint in the checkpoints
        directory and check if there exists the tar file with that
        checkpoint name in order to load the latest weights to continue
        training from that point.

        :param checkpoint_path: Custom directory where checkpoints are saved
        :return: Path of latest checkpoint
        """

        checkpoint_filename = os.path.join(checkpoint_path, "checkpoint")

        if os.path.exists(checkpoint_filename):
            with open(checkpoint_filename) as ckpt_file:
                checkpoints = ckpt_file.read()
            checkpoints=checkpoints.split("\n")
            latest_checkpoint = checkpoints[0]
            latest_checkpoint = latest_checkpoint.split(":")[1]
            latest_checkpoint = latest_checkpoint.strip()
            latest_checkpoint = latest_checkpoint.replace('"', '')
            if os.path.exists(os.path.join(checkpoint_path,latest_checkpoint+".tar")):
                return latest_checkpoint
            else:
                print("tar file for latest checkpoint {} not found in {}".format(latest_checkpoint,checkpoint_path))
                return latest_checkpoint
        else:
            print("checkpoint file that contains name of latest checkpoint not found")
            return None

    def restore_weights(self, checkpoint_path=None, download_from_s3=False, s3_path=None,
                        s3_bucket_name="preeminence-ml-models"):
        """
        Restore weights from the checkpoint path to the latest
        checkpoint to resume training from that point. If download_from_s3
        flag is set, Download the tar file for the latest checkpoint.

        :param checkpoint_path: Path to store downloaded checkpoint and restore weights
        :param download_from_s3: Flag to decide to download checkpoint from s3 or not
        :param s3_path: Path to s3 folder
        :param s3_bucket_name: Name of bucket on s3
        :return: Nothing
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join("./model_weights/",self.name)


        if download_from_s3:
            if s3_path is None:
                raise OSError("s3_path argument not provided.")
            checkpoint_key = os.path.join(s3_path,'checkpoint')

            s3_handler = aws_utils.S3_handler(bucket_name=s3_bucket_name)
            s3_handler.download_file(checkpoint_key,os.path.join(checkpoint_path,"checkpoint"))

            latest_checkpoint = self.get_latest_checkpoint(checkpoint_path=checkpoint_path)

            latest_checkpoint_tar = latest_checkpoint+".tar"

            s3_handler.download_file(os.path.join(s3_path,latest_checkpoint_tar),os.path.join(checkpoint_path,latest_checkpoint_tar))
            utils.untar(os.path.join(checkpoint_path,latest_checkpoint_tar),checkpoint_path)

        if not os.path.exists(checkpoint_path):
            raise OSError("Checkpoint directory not found")
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        # print latest_checkpoint
        if latest_checkpoint:
            if self.saver is None:
                self.saver = tf.train.Saver()
            self.saver.restore(self.sess, latest_checkpoint)
        else:
            raise OSError("Checkpoint file not found at " + checkpoint_path)

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

    def save_weights(self, checkpoint_path=None, checkpoint_number=None, upload_to_s3=False,
                     s3_path=None,s3_bucket_name = "preeminence-ml-models", weight_file_prefix=""):
        """
        Save the weights in current graph to disk in form of
        tar archive. Optionally upload the checkpoint tar to s3

        :param checkpoint_path: Local directory to save checkpoints
        :param checkpoint_number: Number to append to checkpoint
        :param upload_to_s3: Flag to decide to upload checkpoint to s3
        :param s3_path: S3 path to upload checkpoint
        :param s3_bucket_name: S3 bucket name
        :param weight_file_prefix: Optional prefic to weights file
        :return: Nothing
        """

        if checkpoint_path is None:
            print("Checkpoint path not provided. Saving to ./model_weights/")
            checkpoint_path = os.path.join("./model_weights/",self.name)
        if not os.path.exists(checkpoint_path):
            print("Checkpoint path not found. Making directory..")
            os.makedirs(checkpoint_path)

        if checkpoint_number is None:
            latest_checkpoint = self.get_latest_checkpoint(checkpoint_path=checkpoint_path)
            # print latest_checkpoint
            if latest_checkpoint:
                checkpoint_number = int(latest_checkpoint.split("-")[-1]) + 1
            else:
                checkpoint_number = 1

        if self.saver is None:
            self.saver = tf.train.Saver()

        save_path = self.saver.save(self.sess, os.path.join(checkpoint_path,weight_file_prefix), global_step=checkpoint_number)
        save_path = utils.create_checkpoint_tar(save_path)
        print("Model saved at", save_path)
        if upload_to_s3:
            start_time = time.time()
            if s3_path is None:
                raise OSError("s3_path argument not provided.")
            else:

                checkpoint_tar_name = os.path.basename(save_path)
                s3_tar_path = os.path.join(s3_path,self.name,checkpoint_tar_name)


                s3_handler = aws_utils.S3_handler(bucket_name=s3_bucket_name)
                s3_handler.upload_file(save_path,s3_tar_path)
                s3_handler.upload_file(os.path.join(checkpoint_path,"checkpoint"),os.path.join(s3_path,self.name,"checkpoint"))


                print("Model uploaded to S3 at {}/{} in {} seconds".format(s3_bucket_name, s3_path,
                                                                            time.time() - start_time))

