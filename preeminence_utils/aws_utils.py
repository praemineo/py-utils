import boto3
import os

class S3_handler:
    def __init__(self, bucket_name):
        self.s3_client = boto3.client('s3')
        try:
            self.buckets_list = self.get_bucket_names()
        except Exception as e:
            raise AttributeError("Could not fetch bucket list. Please check your internet connection or AWS access keys.")

        if bucket_name in self.buckets_list:
            self.bucket_name = bucket_name
        else:
            raise AttributeError("Bucket with name {} not found on S3".format(bucket_name))

    def get_bucket_names(self):
        response = self.s3_client.list_buckets()
        buckets = [bucket['Name'] for bucket in response['Buckets']]
        return buckets

    def upload_file(self,local_file_path,s3_file_path):
        if os.path.exists(local_file_path):
            self.s3_client.upload_file(local_file_path,self.bucket_name,s3_file_path)
        else:
            raise AttributeError("Provided local file {} does not exist".format(local_file_path))

    def download_file(self,s3_file_path,local_file_path,create_download_path=False):
        if not os.path.exists(os.path.dirname(local_file_path)):
            if create_download_path:
                os.makedirs(os.path.dirname(local_file_path))
            else:
                raise OSError("{} path does not exist. Use create_download_path flag to create this path.")

        self.s3_client.download_file(self.bucket_name, s3_file_path, local_file_path)



