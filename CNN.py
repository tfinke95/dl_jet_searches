from argparse import ArgumentParser
import keras, h5py, time
from keras.layers import *
import numpy as np
import sklearn, time
from scipy.ndimage import gaussian_filter
from classes import *


def loadData(bg_file, sig_file, blurred, indices=slice(40000)):
    """Function to load data from files given as arguments during execution
    input:
      indices: slice object
    returns:
      images: (nImages, 40, 40, 1) jet images
      labels: (nImages, 2) one hot encoded labels
    """
    print(bg_file)
    with h5py.File(bg_file, 'r') as f:
        print(f['images'].shape)
        images = f['images'][indices]
        nBG = len(images)
        labels = np.zeros(nBG)

    print(np.shape(images), np.shape(labels))

    print(sig_file)
    with h5py.File(sig_file, 'r') as f:
        print(f['images'].shape)
        images = np.append(images, f['images'][indices], axis=0)
        nSIG = len(images) - nBG
        labels = np.append(labels, np.ones(nSIG))

    indices = np.arange(len(labels))
    np.random.shuffle(indices)

    labels = labels[indices]
    images = images[indices]

    if blurred:
        images = gaussian_filter(images, 
            sigma=[0, blurred, blurred],
            mode='constant',
            cval=0.0)
        maxes = np.max(images, axis=(1,2))
        maxes = maxes[..., np.newaxis, np.newaxis]
        images /= maxes
        print('blurred')
    images = images[..., np.newaxis]

    labels = keras.utils.to_categorical(labels, 2)
    
    print(images.shape, labels.shape)
    return images, labels


def processInput():
    """Evaluate given flags
    input:
      - Flags and arguments set during execution
    returns:
      args - Namespace object containing set parameters and flags
    """
    parser = ArgumentParser()
    parser.add_argument("-E","--epochs", type=int, default=100,
        help="number of adversarially trained epochs, default: 100")
    parser.add_argument("-b", "--batchsize", type=int, default=512,
        help="set the batch size during training, default: 512")
    parser.add_argument("-n", "--n_jets", type=int, default=40000,
        help="Number of jets per jet type")
    parser.add_argument("--bg_file", 
        help="Give path to file containing background jets")
    parser.add_argument("--sig_file",
        help="Give path to file containing signal jets")
    parser.add_argument("--savetag", 
        help="Give a tag to find saved networks")
    parser.add_argument("--blurred", type=float,
        help="Blur images with given value as sigma")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = processInput()
    start = time.time()

    # Load images and labels from given files
    images, labels = loadData(
                        args.bg_file,
                        args.sig_file,
                        args.blurred,
                        slice(args.n_jets)
                        )

    # Build the classifier
    classi = CNNClassifier(input_shape=(40, 40, 1))

    fit = classi.train(
            images,
            labels,
            args.batchsize,
            args.epochs,
            args.savetag
            )
