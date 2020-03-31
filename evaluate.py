import os
import glob
import time
import argparse
import numpy as np
import h5py

# Kerasa / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from loss import depth_loss_function
from utils import predict, load_images, display_images, evaluate, compare
from matplotlib import pyplot as plt
from PIL import Image

from keras import backend as K
import tensorflow as tf

from keras.utils import multi_gpu_model

from data import NYU_BasicRGBSequence, createBorder
from model import create_model

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
#parser.add_argument('--model', default='../data/nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--model', default='../data/models/r4dl_data2_balanced/model.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--split_eval', default=False, type=bool, help='Split evaluation by ground truth depth value')

def DepthNorm(depth, maxDepth=1000.0): 
    return maxDepth / depth

def load_multigpu_checkpoint_weights(model, h5py_file):
    """
    Loads the weights of a weight checkpoint from a multi-gpu
    keras model.

    Input:

        model - keras model to load weights into

        h5py_file - path to the h5py weights file

    Output:
        None
    """

    print("Setting weights...")
    with h5py.File(h5py_file, "r") as file:

        # Get model subset in file - other layers are empty
        model_file = file["model_weights"]

        try:
            weight_file = model_file["model_1"]
        except:
            weight_file = model_file

        for layer in model.layers:
            print('.', end = '')
            try:
                layer_weights = weight_file[layer.name]

            except:
                # No weights saved for layer
                continue

            try:
                weights = []
                # Extract weights
                for term in layer_weights:
                    if isinstance(layer_weights[term], h5py.Dataset):
                        # Convert weights to numpy array and prepend to list
                        weights.insert(0, np.array(layer_weights[term]))

                # Load weights to model
                layer.set_weights(weights)

            except Exception as e:
                print("Error: Could not load weights for layer:", layer.name)

if __name__ == '__main__':

    args, _ = parser.parse_known_args()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    # Custom object needed for inference and training
    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}

    # Load model into GPU / CPU
    print('Loading model...')

    file, ext = os.path.splitext(args.model) 

    if (ext == '.hdf5'):
        model = create_model()
        load_multigpu_checkpoint_weights(model, args.model)
        model.save(file + '.h5')
    else:
        model = load_model(args.model, custom_objects=custom_objects, compile=False)

    # Load test data
    print('Loading test data...', end='')

    import numpy as np
    from data import extract_zip
    data = extract_zip('../data/nyu_test.zip')
    from io import BytesIO
    rgb = np.load(BytesIO(data['eigen_test_rgb.npy']))
    depth = np.load(BytesIO(data['eigen_test_depth.npy']))
    crop = np.load(BytesIO(data['eigen_test_crop.npy']))

    print('Test data loaded.\n')

    start = time.time()
    print('Testing...')

    if args.split_eval:
        e = evaluate(model, rgb, depth, crop, batch_size=6, scale=True, showImages=False, split_errors=True)

        for bucket in e:
            print(" range: {:10.4f} to {:10.4f}, rmse {:10.4f}, avg num vals per img {:10.4f}".format(bucket[0], bucket[1], bucket[2], bucket[3]/654))

    else:
        e = evaluate(model, rgb, depth, crop, batch_size=6, scale=True, showImages=False)

        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5]))

    end = time.time()
    print('\nTest time', end-start, 's')
