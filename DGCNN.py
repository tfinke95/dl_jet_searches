import keras
import numpy as np 
import tensorflow as tf
from classes import EdgeConv, EdgeConvClassifier
import utils
from argparse import ArgumentParser


def processInput():
    """Parse over the input flags and extract parameters
    Input:
      Script execution from command line
    Returns:
      args: directory containing parameters for program execution
    """
    parser = ArgumentParser()
    parser.add_argument("-E", "--epochs", default=20, type=int, 
        help="Training epochs")
    parser.add_argument("-bg", "--bg_file", help="Path to bg events")
    parser.add_argument("-sig", "--sig_file", help="Path to sigg events")
    parser.add_argument("-n", "--nConst", default=40, type=int,
        help="Number of constituents loaded in the network")
    parser.add_argument("-N", "--nJets", default=100000, type=int,
        help="Number of jets for training")
    parser.add_argument("-S", "--savetag", default='test', type=str,
        help="Give tag for saving files")
    args = parser.parse_args()
    return args


def getCallbacks(savetag):
    """Create callbacks for model training
    Input:
      savetag: str - tag for saving results
    Returns:
      callbacks: list - list of keras callback objects used for training
    """
    def schedule(epoch):
        initial = 3e-4
        lrs = np.linspace(initial, 3e-3, 8)
        lrs = np.append(lrs, lrs[::-1])
        lrs = np.append(lrs, np.linspace(lrs[-1], 5e-7, 4))

        if epoch >= len(lrs):
            return lrs[-1]
        else:
            return lrs[epoch]

    checkpoint = keras.callbacks.ModelCheckpoint(
        'results/networks/best_{}.h5'.format(savetag),
        save_best_only=True, monitor='val_loss')
    lrSchedule = keras.callbacks.LearningRateScheduler(schedule, verbose=0)
    callbacks = [checkpoint, lrSchedule]
    return callbacks


def getData(bg_file, sig_file, nJets, nConst, test=False):
    """Get data for training or testing
    Input:
      bg_file: str - path to file from which to load bg (and sig) data
      sig_file: str or None - path to sig data, if None sig in bg_file
      nJets: int - number of jets loaded
      nConst: int - number of constituents loaded per jet
    Returns:
      momenta: (nJets, nConst, 7) - array with constituent features
      labels: (nJetas, 2) - array with one hot labels
    """
    if sig_file:
        nJets = slice(nJets)
        momenta, labels = utils.loadJetMomenta([bg_file, sig_file],
            nJets=nJets, nConstituents=nConst)
        momenta = np.transpose(momenta, axes=(0, 2, 1))
    else:
        momenta, labels = utils.loadJetMomentaBenchmark(
            bg_file, nConstituents=nConst, end=nJets)
    
    momenta = utils.transformMomenta(momenta)
    return momenta, labels


if __name__ == "__main__":
    args = processInput()
    print("\n\n\nParameters used:\n{}\n".format(args))

    classifier = EdgeConvClassifier((args.nConst, 7))

    # Load momenta and labels from given files
    momenta, labels = getData(
        bg_file=args.bg_file,
        sig_file=args.sig_file,
        nJets=args.nJets,
        nConst=args.nConst)

    # Train the classifier
    classifier.train(
        momenta,
        labels,
        callbacks=getCallbacks(args.savetag),
        epochs=args.epochs,
        savetag=args.savetag
        )
