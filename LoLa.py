import keras, h5py, pandas
import numpy as np
from keras.layers import *
from argparse import ArgumentParser
from classes import LoLaClassifier, CoLa, LoLa
import utils


def processInput():
    """Evaluate given flags
    input:
      - Flags and arguments set during execution
    returns:
      args - Namespace object containing set parameters and flags
    """
    parser = ArgumentParser()
    parser.add_argument("-bg", "--bg_file", 
        help="""Path to the background file.""")
    parser.add_argument("-sig", "--sig_file", 
        help="""Path to the background file.""")
    parser.add_argument("-s", "--savetag", default='test',
        help="""Set a tag that is added to everything that is saved""")
    parser.add_argument("-nC", "--nConst", default=40, type=int,
        help="Number of constituents per jet")
    parser.add_argument("-nA", "--nAdded", default=10, type=int,
        help="Number of added linear combinations in CoLa. Default=10")    
    parser.add_argument("-E", "--epochs", default=100, type=int,
        help="Number of epochs for training")
    parser.add_argument("-N", "--nJets", default=200000, type=int,
        help="Number of training jets per class")
    args = parser.parse_args()
    return args


def loadData(filenames, nJets=slice(200000), nConstituents=40):
    with h5py.File(filenames[0], 'r') as f:
        bg_momenta = f['momenta'][nJets, :nConstituents, :]
        bg_labels = np.zeros(len(bg_momenta))
    
    with h5py.File(filenames[1], 'r') as f:
        sig_momenta = f['momenta'][nJets, :nConstituents, :]
        sig_labels = np.ones(len(sig_momenta))

    momenta = np.append(bg_momenta, sig_momenta, axis=0)
    momenta = np.transpose(momenta, axes=(0, 2, 1))

    labels = keras.utils.to_categorical(
            np.append(bg_labels, sig_labels), 2)
    indices = np.random.permutation(len(labels))
    print(momenta.shape, labels.shape)
    return momenta[indices], labels[indices]


if __name__ == "__main__":
    args = processInput()
    print('\nUsed parameters:\n{}\n'.format(args))

    # Load 4 momenta and labels from given files
    vectors, labels = loadData(
                filenames = [args.bg_file, args.sig_file],
                nJets=slice(args.nJets),
                nConstituents=args.nConst
                )

    # Build classifier
    model = LoLaClassifier(nConstituents=args.nConst, nAdded=args.nAdded).model
    print(model.summary())
    model.compile(
            optimizer=keras.optimizers.Adam(lr=0.001), 
            loss='categorical_crossentropy', 
            metrics=['acc'])

    # Create callbacks for training
    earlystop = keras.callbacks.EarlyStopping(
        patience=8, 
        restore_best_weights=True)
    reduceLR = keras.callbacks.ReduceLROnPlateau(
        patience=3,
        verbose=1,
        min_lr=1e-8)
    model.fit(vectors, labels,
        verbose=2,
        batch_size=512,
        callbacks=[earlystop, reduceLR],
        validation_split=0.1,
        epochs=args.epochs)

    # Save model for further use
    model.save('{}.h5'.format(args.savetag))
