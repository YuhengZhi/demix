import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-9.0/lib64'
print(os.environ['LD_LIBRARY_PATH'])
import tensorflow as tf
# import torch