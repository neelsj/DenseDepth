import os, sys, glob, time, pathlib, argparse, platform, csv, cv2, numpy as np
from PIL import Image
from scipy.interpolate import griddata     
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

# Kerasa / TensorFlow
from loss import depth_loss_function
from utils import predict, save_images, load_test_data
from model import create_model
from data import get_nyu_train_test_data, get_unreal_train_test_data, get_r4dl_train_test_data
from callbacks import get_nyu_callbacks

from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.utils.vis_utils import plot_model

import multiprocessing

import keras
import tensorflow as tf

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--data', default='nyu', type=str, help='Training dataset.')
parser.add_argument('--datadir', default='E:/Source/R4DL/data/', type=str, help='Dataset directory.')
parser.add_argument('--datazip', default='', type=str, help='Dataset zip file.')
parser.add_argument('--datacsv', default='', type=str, help='Dataset csv file.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--bs', type=int, default=4, help='Batch size')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')
parser.add_argument('--gpuids', type=str, default='0', help='IDs of GPUs to use')
parser.add_argument('--mindepth', type=float, default=10.0, help='Minimum of input depths')
parser.add_argument('--maxdepth', type=float, default=1000.0, help='Maximum of input depths')
parser.add_argument('--name', type=str, default='densedepth_nyu', help='A name to attach to the training session')
parser.add_argument('--checkpoint', type=str, default='', help='Start training from an existing model.')
parser.add_argument('--full', dest='full', action='store_true', help='Full training with metrics, checkpoints, and image samples.')
parser.add_argument('--border', type=int, default=8, help='Image border')

def hollFillDepthImages(args, r4dl_data_csv_file='r4dl_path_256_data/R4DL_train.csv', r4dl_data_csv_outfile='r4dl_path_256_data/R4DL_train_proc.csv'):

    with open(args.datadir + r4dl_data_csv_file) as dataFile, open(args.datadir + r4dl_data_csv_outfile, 'w', newline='') as dataFileOut:
        image_data = csv.reader(dataFile)
        writer = csv.writer(dataFileOut, delimiter=',')

        for row in image_data:

            depth = np.asarray(Image.open(args.datadir + row[1])).copy()

            ind = depth==0
            nind = depth!=0

            if (not nind.any() or np.mean(depth) < 1):
                print(row[1] + ' all zero!')
                continue

            rgb = np.asarray(Image.open(args.datadir + row[0])).copy()

            rgbind = rgb==0
            rgbnind = rgb!=0

            if (not rgbnind.any() or np.mean(rgb) < 1):
                print(row[0] + ' all zero!')
                continue

            if (ind.any()): #fill is any invalid pixels
                
                print(row[1])

                xx, yy = np.meshgrid(range(depth.shape[1]),range(depth.shape[0]))
                depth[ind] = griddata((xx[nind], yy[nind]), depth[nind], (xx[ind], yy[ind]), method='nearest')

                Image.fromarray(depth).save(args.datadir + row[1])

            writer.writerow(row)
            dataFileOut.flush()

def main(args):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    keras.backend.set_session(tf.Session(config=config))

    threads = min(max(args.bs,8), multiprocessing.cpu_count())

    # Inform about multi-gpu training
    if args.gpus == 1: 
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuids
        print('Will use GPU ' + args.gpuids)
    else:
        print('Will use ' + str(args.gpus) + ' GPUs.')

    # Create the model
    model_file = (args.datadir + args.checkpoint) if (len(args.checkpoint) != 0) else ''
    model = create_model(existing = model_file)

    # Data loaders
    if args.data == 'nyu': train_generator, test_generator = get_nyu_train_test_data( args.bs,  args.datadir, nyu_data_train_csv=args.datacsv)
    if args.data == 'unreal': train_generator, test_generator = get_unreal_train_test_data( args.bs, args.datadir )
    if args.data == 'r4dl': train_generator, test_generator = get_r4dl_train_test_data( args.bs, args.datadir, r4dl_data_csv_file=args.datacsv)

    # Training session details
    runID = str(int(time.time())) + '-n' + str(len(train_generator)) + '-e' + str(args.epochs) + '-bs' + str(args.bs) + '-lr' + str(args.lr) + '-' + args.name
    outputPath = args.datadir + 'models/'
    runPath = outputPath + runID
    pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
    print('Output: ' + runPath)

     # (optional steps)
    if True:
        # Keep a copy of this training script and calling arguments
        with open(__file__, 'r') as training_script: training_script_content = training_script.read()
        training_script_content = '#' + str(sys.argv) + '\n' + training_script_content
        with open(runPath+'/'+os.path.basename(__file__), 'w') as training_script: training_script.write(training_script_content)

        # Generate model plot
         
        if(platform.system() != 'Windows'):
            plot_model(model, to_file=runPath+'/model_plot.svg', show_shapes=True, show_layer_names=True)

        # Save model summary to file
        from contextlib import redirect_stdout
        with open(runPath+'/model_summary.txt', 'w') as f:
            with redirect_stdout(f): model.summary()

    # Multi-gpu setup:
    basemodel = model
    if args.gpus > 1: model = multi_gpu_model(model, gpus=args.gpus)

    # Optimizer
    optimizer = Adam(lr=args.lr, amsgrad=True)

    # Compile the model
    print('\n\n\n', 'Compiling model..', runID, '\n\n\tGPU ' + (str(args.gpus)+' gpus' if args.gpus > 1 else args.gpuids)
            + '\t\tBatch size [ ' + str(args.bs) + ' ] ' + ' \n\n')
    model.compile(loss=depth_loss_function, optimizer=optimizer)

    print('Ready for training using %d GPU and %d CPU threads!\n' % (args.gpus, threads))

    # Callbacks
    callbacks = get_nyu_callbacks(model, basemodel, train_generator, test_generator, load_test_data(args.datadir) if args.full else None , runPath)

    # Start training
    model.fit_generator(train_generator, callbacks=callbacks, validation_data=test_generator, epochs=args.epochs, shuffle=True, workers=threads)

    # Save the final trained model:
    basemodel.save(runPath + '/model.h5')

if __name__ == '__main__':

    args = parser.parse_args()

    if ('PT_DATA_DIR' in os.environ):
        args.datadir = os.environ['PT_DATA_DIR'] + '/'

    main(args)
    #hollFillDepthImages(args, 'fixed_path_256_16_jitters_training_data/R4DL_train.csv', 'fixed_path_256_16_jitters_training_data/R4DL_train_proc2.csv')
    

