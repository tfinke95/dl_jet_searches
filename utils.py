import h5py
import numpy as np
import keras, pandas
from sklearn import metrics


def loadJetMomenta(filenames, nJets=slice(200000), nConstituents=40):
    """Load jet momenta from given files
    Input:
     filenames: (2,) - list containing path to bg and sig jet momenta
     nJets: slice - indices for loading data
     nConstituents: int giving number of leading pT constituents per jet
    Returns:
     momenta: (nJets, 4, nConstituents) shuffled array of jet momenta
     labels: (nJets, 2) one hot encoded labels for momenta
    """
    print("Loading by given slice {}".format(nJets))
    # Load momenta from bg_file
    with h5py.File(filenames[0], 'r') as f:
        if f['momenta'].shape[-1] == 4:
            bg_momenta = f['momenta'][nJets, :nConstituents, :]
        elif f['momenta'].shape[-2] == 4:
            bg_momenta = f['momenta'][nJets, :, :nConstituents]
            bg_momenta = np.transpose(bg_momenta, (0, 2, 1))
        else:
            print("Momenta cannot be loaded")
            exit()
        bg_labels = np.zeros(len(bg_momenta))
        print("\n{} bg jets loaded from {}".format(
            len(bg_labels), filenames[0]))

    # Load momenta from sig_file
    with h5py.File(filenames[1], 'r') as f:
        if f['momenta'].shape[-1] == 4:
            sig_momenta = f['momenta'][nJets, :nConstituents, :]
        elif f['momenta'].shape[-2] == 4:
            sig_momenta = f['momenta'][nJets, :, :nConstituents]
            sig_momenta = np.transpose(sig_momenta, (0, 2, 1))
        else:
            print("Momenta cannot be loaded")
            exit()
            
        sig_labels = np.ones(len(sig_momenta))
        print("{} sig jets loaded from {}\n\n".format(
            len(sig_labels), filenames[1]))


    # Connect momenta, shuffle and return them with corresponding labels
    momenta = np.append(bg_momenta, sig_momenta, axis=0)
    labels = keras.utils.to_categorical(
            np.append(bg_labels, sig_labels), 2)
    indices = np.random.permutation(len(labels))
    print(momenta.shape, labels.shape)
    return momenta[indices], labels[indices]


def loadJetMomentaBenchmark(filename, nConstituents=40, start=None, end=100000):
    """Load jet momenta from top tagging dataset as used in arXiv:1902.09914
    Input:
     filename: str, path to file
     nConstituents: int, number of constituents loaded for each jet
     start: int, index from which to start select events
     end: int, index up to which select events
    Returns:
     momenta: (nJets, 4, nConstituents), array with 4 momenta of jet constituents
     labels: (nJets, 2), one-hot encoded labels for jets
    """

    with pandas.HDFStore(filename, 'r') as store:
        print("Loading indices from {} to {}".format(start, end))
        events = store.select("table", start=start, stop=end)
        momenta = events.values[:, :nConstituents*4]
        momenta = np.reshape(momenta, (len(momenta), nConstituents, 4))
        momenta = np.transpose(momenta, [0, 2, 1])
        labels = events.values[:, -1]
        print('\nMean label\n{}\n'.format(np.mean(labels)))
        labels = keras.utils.to_categorical(labels, 2)

    indices = np.random.permutation(len(labels))
    return momenta[indices], labels[indices]


def transformMomenta(momenta):
    """Transform 4 momenta into vector with delta eta, delta phi, log pT, log E,
    log pT / pTjet, log E / Ejet, delta R
    Input:
     momenta: (nJets, 4, nConstituents)
    Returns:
     momenta: (nJets, nConstituents, 7)
    """
    # Get jet properties
    jetMomenta = np.sum(momenta, axis=2)
    jetPt = np.linalg.norm(jetMomenta[:, 1:3], axis=1)[..., np.newaxis]
    jetE = jetMomenta[:, 0][..., np.newaxis]
    jetP = np.linalg.norm(jetMomenta[:, 1:], axis=1)
    jetEta = 0.5 * np.log((jetP + jetMomenta[:, 3]) / (jetP - jetMomenta[:, 3]))[..., np.newaxis]
    jetPhi = np.arctan2(jetMomenta[:, 2], jetMomenta[:, 1])[..., np.newaxis]

    # Get pT, E, p, eta, phi and dR for all constituents
    pT = np.linalg.norm(momenta[:, 1:3, :], axis=1)
    e = momenta[:, 0, :]
    p = np.linalg.norm(momenta[:, 1:, :], axis=1)
    eta = 0.5 * np.log((p + momenta[:, 3, :]) / (p - momenta[:, 3, :]))
    eta -= jetEta
    phi = np.arctan2(momenta[:, 2, :], momenta[:, 1, :])
    phi = np.unwrap(phi - jetPhi)
    dR = np.sqrt(phi ** 2 + eta ** 2)

    # Set calculated features of non-particle entries to zero
    eta[pT==0] = 0
    phi[pT==0] = 0
    dR[pT==0] = 0

    # Stack a new feature vector
    newVec = np.stack([eta, phi, np.log(pT), np.log(e),
        np.log(pT / jetPt), np.log(e / jetE), dR], axis=-1)
    newVec[newVec==-np.inf] = 0  # Deal with infinities
    
    return newVec


if __name__ == "__main__":
    pass
