
from setuptools import setup

setup(name='preeminence_utils',
      version='0.1',
      description='Package for Internal usage in Preeminence.',
      url='https://github.com/preeminence/py-utils',
      author='Tushar Pawar',
      author_email='tusharvijaypawar@gmail.com',
      license='MIT',
      packages=['preeminence_utils'],
      install_requires=[
          'pymongo',
      ],
      zip_safe=False)