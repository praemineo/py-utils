
from setuptools import setup,find_packages

setup(name='preeminence_utils',
      version='0.2',
      description='Package for Internal usage in Preeminence.',
      url='https://github.com/preeminence/py-utils',
      author='Tushar Pawar',
      author_email='tusharvijaypawar@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[


          'pymongo','numpy','awscli','boto3'

      ],
      zip_safe=False)