import keras, h5py, time, sys
from keras.layers.advanced_activations import PReLU
import numpy as np
import keras.backend as K
from argparse import ArgumentParser
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from classes import *


def loadData(nJets=550000, percentage=0):
    """Load training images and corresponding jet masses
    input:
      nJets: int - Number of jets loaded for training
      percentage: float - amount of signal jets in the total training set
    returns:
      masses: (nJets) - Jet masses
      images: (nJets, 40, 40, 1) - Jet images
    """
    
    def prepareMasses(masses):
        """Function gets a list of masses and sorts them into 10 equally 
        populated bins.
        input:
          masses: (nJets) - List of jet masses
        returns:
          masses_categories: (nJets) - List of int giving the massbin
        """
        temp_masses = sorted(masses)
        massbins = 10
        step = len(temp_masses) // massbins
        binedges = [0.0]
        for i in range(1, massbins):
            binedges.append(temp_masses[i * step - 1])

        masses_categories = np.digitize(masses, binedges, right=True)

        return np.array(masses_categories)


    # Calculate number of signal jets in training set
    nJets_sig = int(percentage * nJets)
    nJets_bg = nJets - nJets_sig
    print('Signal jets: {}\nBackground jets: {}'.format(nJets_sig, nJets_bg))

    # Load BG jets
    with h5py.File(args.bg_file, 'r') as f:
        print('\nLoading background jets from {}'.format(args.bg_file))
        print(f['masses'].shape)
        masses = f['masses'][:nJets_bg]
        images = f['images'][:nJets_bg]
    
    print(np.shape(images), np.shape(masses))

    # Load signal jets, if there need to be any
    if nJets_sig != 0:
        print("\nLoading signal jets\n")
        with h5py.File(args.sig_file, 'r') as f:
            print(f['masses'].shape)
            masses = np.append(masses, f['masses'][:nJets_sig], axis=0)
            images = np.append(images, f['images'][:nJets_sig], axis=0)
    
    images = images[..., np.newaxis]
    
    # Prepare one hot encoded mass labels
    masses = prepareMasses(masses)
    masses = keras.utils.to_categorical(masses, num_classes=12)  
    
    indices = np.arange(0, len(images))
    np.random.shuffle(indices)
    images = images[indices]
    masses = masses[indices]
    
    print(np.shape(images), np.shape(masses))
    return masses, images


def processInput():
    """Parse over the input flags and extract parameters
    Input:
      Script execution from command line
    Returns:
      args: directory containing parameters for program execution
    """
    parser = ArgumentParser()
    parser.add_argument("-N","--nJets", type=int, default=100000,
        help="set total number of training jets, default: 100000")
    parser.add_argument("-E","--epochs", type=int, default=100,
        help="number of adversarially trained epochs, default: 100")
    parser.add_argument("-b", "--batchsize", type=int, default=512,
        help="set the batch size during training, default: 512")
    parser.add_argument("-w", "--weight", type=float, default=3e-4,
        help="weightfactor of the adversarial loss, default: 3e-4")
    parser.add_argument("-m", "--mixing", type=float, default=0.0,
        help="percentage of top jets in mixed training, default: 0.0")
    parser.add_argument("-s", "--savefreq", type=int, default=50,
        help="set number of epochs after which network is saved, default: 50")
    parser.add_argument("-r", "--redfreq", type=int, default=10,
        help="set number of epochs after which lr is reduced, default: 10")
    parser.add_argument("--reduction", type=float, default=0.5,
        help="factor multiplied to LR every redfreq epochs, default: 0.5")
    parser.add_argument("--bg_file", 
        help="Give path to file containing background jets")
    parser.add_argument("--sig_file",
        help="Give path to file containing signal jets")
    parser.add_argument("--savetag", default='test',
        help="Give a tag to find saved networks")
    parser.add_argument("--only_AE", action="store_true", 
        help="If set, train only the autoencoder, without adversarial")
    parser.add_argument("--blurred", action="store_true",
        help="Use gaussian filter over the images to support neighborhoods")
    parser.add_argument("--bneck", type=int, default=32,
        help="Bottleneck size for AE")

    args = parser.parse_args()

    print("""
    ------------------------------------------------
    {}
    ------------------------------------------------""".format(args.savetag))
    return args


if __name__ == "__main__":
    start = time.time()
    
    args = processInput()

    print('\n\n{}\n\n'.format(args))

    masses, images = loadData(nJets=args.nJets, percentage=args.mixing)

    # Apply Gaussian filter, if flag is set
    if args.blurred:
        print('\nBlurred images\n')
        images = gaussian_filter(images[:, :, :, 0], sigma=[0, 5, 5])
        maxes = np.max(images, axis=(1,2))
        maxes = maxes[..., np.newaxis, np.newaxis]
        images /= maxes
        images = images[..., np.newaxis] 

    # Train the AE, if only AE is set
    if args.only_AE:
        print("\nBegin AE training\n")
        
        ae = Autoencoder(bottleneck=args.bneck).AE
        print(ae.summary())

        opt = keras.optimizers.Adam(lr=0.001)
        ae.compile(optimizer=opt, loss='mse')

        earlyStop = keras.callbacks.EarlyStopping(
            patience=15, 
            restore_best_weights=True)
        reduction = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=5,
            factor=0.5,
            verbose=1,
            min_delta=0,
            min_lr=1e-8)
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath="AE_" + args.savetag + "_epoch{epoch:02d}.h5",
            period=5)

        fit = ae.fit(images, images, 
                batch_size=args.batchsize, epochs=args.epochs, 
                validation_split=0.1, 
                verbose=2,
                callbacks=[
                        reduction,
                        earlyStop])

        ae.save('{}.h5'.format(args.savetag))
    
    else:
        # Create AAE object
        aae = AdversarialAutoencoder(inputshape=(40, 40, 1),
            savetag=args.savetag,
            bottleneck=args.bneck,
            weightfactor=args.weight)
        print("\nBegin AAE training\n")
        aae.alternatingTraining(images, masses,
                batchsize=args.batchsize, epochs=args.epochs, 
                savefreq=args.savefreq, red_freq=args.redfreq,
                reduction=args.reduction)
    
    print('Time needed: {}'.format(int(time.time()-start)))
