import os
import glob
import time
import argparse
import numpy as np
import h5py
from tqdm import tqdm
import csv

from utils import predict, load_images, display_images, evaluate, compare
from matplotlib import pyplot as plt
from PIL import Image

from scipy.stats import entropy

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

if __name__ == '__main__':

    showEachPlot = False
    showFinalPlot = True

    # Load test data
    print('Loading data...')

    datafile = '../data/fixed_path_256_16_jitters_training_data/R4DL_train_proc2.csv'
    #datafile = '../data/data/nyu2_train_512_img_4_scenes.csv'
    #datafile = '../data/data/nyu2_train.csv'
    #datafile = '../data/data/nyu2_test.csv'

    depthScale = 1/255
    # use scale below if loading nyu2_test.csv
    #depthScale = 1/10000
  
    with open(datafile, encoding="utf-8", newline="\n") as test_csv:
        test_data = list((row.strip("\n").split(',') for row in (test_csv) if len(row) > 0))

    test_data = tqdm(test_data)

    #target_hist = None
    target_hist = np.load('../data/data/nyu2_test_total_hist.npy')

    num_bins = 50

    vmin = 1e10
    vmax = 0
    total_hist = None
    avg_ent = 0

    stats = []

    for sample in test_data:

        rgb_path = '../data/' + sample[0].rstrip()
        gt_path = '../data/' + sample[1].rstrip()
        
        rgb = np.array(Image.open( rgb_path)).reshape(480,640,3)
        depth = np.array(Image.open( gt_path)).reshape(480,640)
            
        depth = depth*depthScale

        vmin = min(np.min(depth), vmin)
        vmax = max(np.max(depth), vmax)

        hist,bins_edges = np.histogram(depth, num_bins, (0,1))
        hist = hist/np.sum(hist)
        bin_centers = (bins_edges[0:-1] + bins_edges[1:])/2

        ent = entropy(hist)
        avg_ent += ent

        row = [sample[0].rstrip(), sample[1].rstrip(), ent]

        if (target_hist is not None):
            kld = kl_divergence(np.clip(hist,1e-10, 1), np.clip(target_hist,1e-10, 1))
            row.append(kld)

        stats.append(row)

        # add intro aggregate historgram
        if (total_hist is None):
            total_hist = hist
        else:
            total_hist += hist

        if (showEachPlot):            
            plt.bar(bin_centers,hist,width=1/num_bins)
            plt.show(block=False)
            plt.pause(.001)            

    total_hist /= np.sum(total_hist)            
    avg_ent /= len(test_data)

    if (target_hist is not None):
        total_kld = kl_divergence(np.clip(total_hist,1e-10, 1), np.clip(target_hist,1e-10, 1))

        print('\nvmin %f vmax %f avg ent %f total kld %f \n' % (vmin, vmax, avg_ent, total_kld))
    else:
        print('\nvmin %f vmax %f avg ent %f\n' % (vmin, vmax, avg_ent))
  
    np.save(os.path.splitext(datafile)[0] + '_total_hist', total_hist)

    with open(os.path.splitext(datafile)[0] + '_stats.csv', 'w', newline='') as out_csv:
        writer = csv.writer(out_csv, delimiter=',')

        for row in stats:
            writer.writerow(row)
            out_csv.flush()

    if (showFinalPlot):

        plt.bar(bin_centers,total_hist,width=1/num_bins)
        plt.show()
