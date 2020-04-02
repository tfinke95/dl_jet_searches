from keras.layers import *
from keras import Model
from keras.models import Sequential
from keras.layers.advanced_activations import PReLU
from keras import regularizers
import numpy as np
import keras, time
import keras.backend as K
import tensorflow as tf
import utils
import matplotlib.pyplot as plt


"""-----------------------------------------------------------------------------
Custom layer classes
CoLa -> Combination layer adding linear combinations to list of 4 vectors
        arXiv:1707.08966
LoLa -> Lorentz layer similar to arXiv:1707.08966
EdgeConv -> DGNN convolution, see arXiv:1801.07829
-----------------------------------------------------------------------------"""
class CoLa(Layer):
    """Combination layer that returns the input and appends some linear 
    combinations along the last dimension
    """
    def __init__(self, nAdded, **kwargs):
        # Set the number of added combinations
        self.nAdded = nAdded
        super(CoLa, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create trainable weight for the linear combination
        self.combination = self.add_weight(name='combination',
                    shape=(input_shape[2], self.nAdded),
                    initializer='uniform',
                    trainable=True)
        super(CoLa, self).build(input_shape)

    def call(self, x):
        # Generate combinations and return input with appended with combinations
        combined = K.dot(x, self.combination)
        return K.concatenate([x, combined], axis=2)

    def compute_output_shape(self, input_shape):
        self.out_shape = (input_shape[0], 
                    input_shape[1], 
                    input_shape[2] + self.nAdded)
        return self.out_shape

    def get_config(self):
        # Store nAdded value for loading saved models later
        base_config = super(CoLa, self).get_config()
        base_config['nAdded'] = self.nAdded
        return base_config


class LoLa(Layer):
    """Lorentz Layer adapted from arXiv:1707.08966
    From an input of 4 vectors generate some physical quantities that serve as
    input to a classifiction network
    """
    def __init__(self, **kwargs):
        super(LoLa, self).__init__(**kwargs)

    def build(self, input_shape):
        initializer = keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
        metric = keras.initializers.Constant(value=[[1., -1., -1., -1.]])
        # Trainable metric for 4-vector multiplication
        self.metric = self.add_weight(name='metric',
                    shape=(1, 4),
                    initializer=metric,
                    trainable=True)

        # Weights for the linear combination of energies
        self.energyCombination = self.add_weight(name='energyCombination',
                    shape=(input_shape[-1], input_shape[-1]),
                    initializer=initializer,
                    trainable=True)
        
        # Weights for the linear combinations of distances
        self.distanceCombination = self.add_weight(name='distanceCombination',
                    shape=(input_shape[2], 4),
                    initializer=initializer,
                    trainable=True)
        super(LoLa, self).build(input_shape)

    def call(self, x):
        def getDistanceMatrix(x):
            """Input:
            x, (batchsize, features, nConst) - array of vectors
            Returns:
            dists, (batchsize, nConst, nConst) - distance array for every jet
            """
            part1 = -2 * K.batch_dot(x, K.permute_dimensions(x, (0, 2, 1)))
            part2 = K.permute_dimensions(K.expand_dims(K.sum(x**2, axis=2)), (0, 2, 1))
            part3 = K.expand_dims(K.sum(x**2, axis=2))
            dists = part1 + part2 + part3
            return dists

        # Get mass of each 4-momentum
        mass = K.dot(self.metric, K.square(x))
        mass = K.permute_dimensions(mass, (1, 0, 2))

        # Get pT of each 4-momentum
        pT = x[:, 1, :] ** 2 + x[:, 2, :] ** 2
        pT = K.sqrt(K.reshape(pT, (K.shape(pT)[0], 1, K.shape(pT)[1])))
        
        # Get a learnable linear combination of the energies of all constituents
        energies = K.dot(x[:, 0, :], self.energyCombination)
        energies = K.reshape(energies, 
                            (K.shape(energies)[0], 1, K.shape(energies)[1]))
 
        # Get the distance matrix and do some linear combination
        dists_3 = getDistanceMatrix(
                            K.permute_dimensions(x[:, 1:, :], (0, 2, 1)))
        dists_0 = getDistanceMatrix(
                            K.permute_dimensions(x[:, 0, None, :], (0, 2, 1)))
        dists = dists_0 - dists_3
        
        dists = K.dot(dists, self.distanceCombination)
        dists = K.permute_dimensions(dists, (0, 2, 1))

        return K.concatenate([mass, pT, energies, dists], axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 7, input_shape[2])


class EdgeConv(Layer):
    """ Keras layer to perform EdgeConvolutions (1801.07829)
    From a point cloud as input generate graph with connections between k
    nearest neighbors and perform a convolution over these local patches
    """
    def __init__(self, k=10, kernel_size=1, n_channel_out=[16], 
            nearest_ind=False, **kwargs):
        # Possibility to calculate proximity in given dimensions
        self.n_ind = nearest_ind
        self.k = k
        self.kernel_size = kernel_size
        self.n_channel_out = n_channel_out
        super(EdgeConv, self).__init__(**kwargs)

    def build(self, input_shape):
        n_channel_in = input_shape[-1] * 2
        kernel_shape = [self.kernel_size, self.kernel_size,
                        n_channel_in, self.n_channel_out[0]]

        self.kernel = [self.add_weight(name='kernel0',
                                    shape=kernel_shape,
                                    initializer='truncated_normal',
                                    trainable=True)]
        self.bias = [self.add_weight(name='bias0',
                                    shape=[kernel_shape[-1]],
                                    initializer='zeros',
                                    trainable=True)]
        self.gamma = [self.add_weight(name='gamma0',
                                    shape=[kernel_shape[-1]],
                                    initializer='ones',
                                    trainable=True)]
        self.beta = [self.add_weight(name='beta0',
                                    shape=[kernel_shape[-1]],
                                    initializer='zeros',
                                    trainable=True)]
        self.moving_mean = [self.add_weight(name='moving_mean0',
                                    shape=[kernel_shape[-1]],
                                    initializer='zero',
                                    trainable=False)]
        self.moving_var = [self.add_weight(name='moving_var0',
                                    shape=[kernel_shape[-1]],
                                    initializer='one',
                                    trainable=False)]

        #Adding weights for muore convolutions before constructing a new graph
        for i in range(1, len(self.n_channel_out)):
            kernel_shape = [self.kernel_size, self.kernel_size,
                self.n_channel_out[i-1], self.n_channel_out[i]]
            self.kernel.append(self.add_weight(name='kernel{}'.format(i),
                                    shape=kernel_shape,
                                    initializer='truncated_normal',
                                    trainable=True))
            self.bias.append(self.add_weight(name='bias{}'.format(i),
                                    shape=[kernel_shape[-1]],
                                    initializer='zeros',
                                    trainable=True))
            self.gamma.append(self.add_weight(name='gamma{}'.format(i),
                                    shape=[kernel_shape[-1]],
                                    initializer='ones',
                                    trainable=True))
            self.beta.append(self.add_weight(name='beta{}'.format(i),
                                    shape=[kernel_shape[-1]],
                                    initializer='zeros',
                                    trainable=True))
            self.moving_mean.append(self.add_weight(
                                    name='moving_mean{}'.format(i),
                                    shape=[kernel_shape[-1]],
                                    initializer='zero',
                                    trainable=False))
            self.moving_var.append(self.add_weight(
                                    name='moving_var{}'.format(i),
                                    shape=[kernel_shape[-1]],
                                    initializer='one',
                                    trainable=False))
        super(EdgeConv, self).build(input_shape)

    def call(self, point_cloud):
        def getDistanceMatrix(x):
            """ Compute pairwise distance matrix for a point cloud

            Input:
            point_cloud: tensor (batch_size, n_points, n_features)
            
            Returns:
            dists: tensor (batch_size, n_points, n_points) pairwise distances
            """
            part1 = -2 * K.batch_dot(x, K.permute_dimensions(x, (0, 2, 1)))
            part2 = K.permute_dimensions(K.expand_dims(K.sum(x**2, axis=2)), (0, 2, 1))
            part3 = K.expand_dims(K.sum(x**2, axis=2))
            dists = part1 + part2 + part3
            return dists

        def getKnearest(dists, k):
            """Get indices of k nearest neighbors from distance tensor
            Input:
            dists: (batch_size, n_points, n_points) pairwise distances
            Returns:
            knn_idx: (batch_size, n_points, k) nearest neighbor indices
            """
            _, knn_idx = tf.math.top_k(-dists, k=k)
            return knn_idx

        def getEdgeFeature(point_cloud, nn_idx):
            """Construct the input for the edge convolution
            Input:
            point_cloud: (batch_size, n_points, n_features)
            nn_idx: (batch_size, n_points, n_neighbors)
            Returns:
            edge_features: (batch_size, n_points, k, n_features*2)
            """
            k = nn_idx.get_shape()[-1]

            point_cloud_shape = tf.shape(point_cloud)
            batch_size = point_cloud_shape[0]
            n_points = point_cloud_shape[1]
            n_features = point_cloud_shape[2]

            # Prepare indices to match neighbors in flattened cloud
            idx = K.arange(0, stop=batch_size, step=1) * n_points
            idx = K.reshape(idx, [-1, 1, 1])

            # Flatten cloud and gather neighbors
            flat_cloud = K.reshape(point_cloud, [-1, n_features])
            neighbors = K.gather(flat_cloud, nn_idx+idx)

            # Expand centers to (batch_size, n_points, k, n_features)
            cloud_centers = K.expand_dims(point_cloud, axis=-2)
            cloud_centers = K.tile(cloud_centers, [1, 1, k, 1])

            edge_features = K.concatenate([cloud_centers, neighbors-cloud_centers], 
                axis=-1)
            return edge_features

        def batch_norm(inputs, gamma, beta, dims, ind):
            """ Normalize batch and update moving averages for mean and std
            Input:
              inputs: (batchsize, n_points, k, n_features * 2) - edge_features
              gamma: weight - gamma for batch normalization
              beta: weight - beta for batch normalization
              dims: list - dimensions along which to normalize
              ind: int - indicating which weights to use
            Returns:
             During training:
              normed: (batchsize, n_points, k, n_features * 2) - normalized
                            batch of data using actual batch for normalization
             Else:
              normed_moving: same, but using the updated average values
            """

            # Calculate normalized data, mean and std for batch
            normed, batch_mean, batch_var = K.normalize_batch_in_training(
                                                x=inputs,
                                                gamma=gamma,
                                                beta=beta,
                                                reduction_axes=dims)

            # Update the moving averages
            self.add_update([
                K.moving_average_update(self.moving_mean[ind], batch_mean, 0.9),
                K.moving_average_update(self.moving_var[ind], batch_var, 0.9)])

            # Calculate normalization using the averages
            normed_moving = K.batch_normalization(
                                                x=inputs,
                                                mean=self.moving_mean[ind],
                                                var=self.moving_var[ind],
                                                beta=beta,
                                                gamma=gamma)

            # If training return normed, else normed_moving
            return K.in_train_phase(normed, normed_moving)


        if self.n_ind:  # get dinstances according to given indices
            dists = getDistanceMatrix(
                        point_cloud[:, :, slice(self.n_ind[0], self.n_ind[1])])
        else:  # get distances according to full feature vector
            dists = getDistanceMatrix(point_cloud)

        knn_idx = getKnearest(dists, self.k)
        edge_features = getEdgeFeature(point_cloud, knn_idx)

        # Create first convolutional block
        output = K.conv2d(edge_features, self.kernel[0], (1, 1), padding='same')
        output = K.bias_add(output, self.bias[0])
        output = batch_norm(output, self.gamma[0], self.beta[0], [0, 1, 2], 0)
        output = K.relu(output)

        # Additional convolutional blocks
        for i in range(1, len(self.n_channel_out)):
            output = K.conv2d(output, self.kernel[i], (1, 1), padding='same')
            output = K.bias_add(output, self.bias[i])
            output = batch_norm(output,
                                    self.gamma[i], self.beta[i], [0, 1, 2], i)
            output = K.relu(output)

        output = K.mean(output, axis=-2)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.n_channel_out[-1])

    def get_config(self):
        # Store values necessary to load model later
        base_config = super(EdgeConv, self).get_config()
        base_config['k'] = self.k
        base_config['kernel_size'] = self.kernel_size
        base_config['n_channel_out'] = self.n_channel_out
        base_config['nearest_ind'] = self.n_ind
        return base_config

"""-----------------------------------------------------------------------------
Network architectures used
-----------------------------------------------------------------------------"""
class Autoencoder():
    def __init__(self, input_shape=(40, 40, 1), bottleneck=32):
        self.bneck = bottleneck
        self.input_shape = input_shape
        self._genAE()


    def _genAE(self):
        input = Input(shape=self.input_shape)
        encoded = Conv2D(10, 3, padding='same')(input)
        encoded = PReLU()(encoded)
        encoded = Conv2D(10, 3, padding='same')(encoded)
        encoded = PReLU()(encoded)
        encoded = AveragePooling2D()(encoded)
        encoded = Conv2D(10, 3, padding='same')(encoded)
        encoded = PReLU()(encoded)
        encoded = Conv2D(5, 3, padding='same')(encoded)
        encoded = PReLU()(encoded)
        encoded = Conv2D(5, 3, padding='same')(encoded)
        encoded = PReLU()(encoded)
        encoded = Flatten()(encoded)
        encoded = Dense(400)(encoded)
        encoded = PReLU()(encoded)
        encoded = Dense(100)(encoded)
        encoded = PReLU()(encoded)
        encoded = Dense(self.bneck)(encoded)
        encoded = PReLU(name='bneck')(encoded)
        
        decoded = Dense(100)(encoded)
        decoded = PReLU()(decoded)
        decoded = Dense(400)(decoded)
        decoded = PReLU()(decoded)
        decoded = Reshape((20, 20, 1))(decoded)
        decoded = Conv2D(5, 3, padding='same')(decoded)
        decoded = PReLU()(decoded)
        decoded = Conv2D(5, 3, padding='same')(decoded)
        decoded = PReLU()(decoded)
        decoded = UpSampling2D()(decoded)
        decoded = Conv2D(5, 3, padding='same')(decoded)
        decoded = PReLU()(decoded)
        decoded = Conv2D(10, 3, padding='same')(decoded)
        decoded = PReLU()(decoded)
        decoded = Conv2D(1, 3, padding='same')(decoded)
    
        self.AE = Model(input, decoded, name='heimelAE')


class CNNClassifier():
    def __init__(self, input_shape=(40, 40, 1)):
        self.img_shape = input_shape
        self._genNetwork()
        self.model.compile(optimizer=keras.optimizers.Adam(lr=0.001), 
                loss='binary_crossentropy',
                metrics=['acc'])


    def _genNetwork(self):
        input_img = Input(shape=self.img_shape)
        layer = input_img
        layer = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                            name='conv1')(layer)
        layer = MaxPooling2D(pool_size=(2,2), padding='same')(layer)
        layer = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                            name='conv2')(layer)
        layer = MaxPooling2D(pool_size=(2,2), padding='same')(layer)
        layer = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                            name='conv3')(layer)
        layer = Flatten()(layer)
        layer = Dense(32, activation='relu', name='dense1')(layer)
        layer = Dense(6, name='dense2')(layer)
        
        classes = Dense(2, activation='softmax', name='classes')(layer)

        self.model = keras.Model(input_img, classes)
        print(self.model.summary())


    def train(self, images, labels, batchsize=512, epochs=100, savetag='test'):
        earlystopping = keras.callbacks.EarlyStopping(patience=8, 
                                            restore_best_weights=True,
                                            verbose=1)
        red = keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.1,
                                            verbose=1)
        fit = self.model.fit(images, labels,
                batch_size=batchsize, epochs=epochs, verbose=2,
                validation_split=0.1,
                # class_weight={0: 1., 1: 1.},
                callbacks=[earlystopping, red])
        self.model.save("{}.h5".format(
            savetag))
        return fit


class AdversarialAutoencoder():
    def __init__(self, bottleneck, inputshape, weightfactor, savetag):        
        self.bottleneck = bottleneck
        self.inputshape = inputshape
        self.optimizerAE = keras.optimizers.Adam(lr=0.001)
        self.optimizerAdversarial = keras.optimizers.Adam(lr=0.01)
        self.weightfactor = weightfactor
        self.timestamp = savetag

        print('\n----------------------------------------------------\n')
        print(self.timestamp)
        print('\n----------------------------------------------------\n')
        
        self._genAE()
        self._genAdversarial()
        self._genAAE()


    # Generate the autoencoder model
    def _genAE(self):
        input = keras.layers.Input(shape=self.inputshape)
        encoded = keras.layers.Conv2D(10, 3, padding='same')(input)
        encoded = PReLU()(encoded)
        encoded = keras.layers.Conv2D(10, 3, padding='same')(encoded)
        encoded = PReLU()(encoded)
        encoded = keras.layers.AveragePooling2D()(encoded)
        encoded = keras.layers.Conv2D(10, 3, padding='same')(encoded)
        encoded = PReLU()(encoded)
        encoded = keras.layers.Conv2D(5, 3, padding='same')(encoded)
        encoded = PReLU()(encoded)
        encoded = keras.layers.Conv2D(5, 3, padding='same')(encoded)
        encoded = PReLU()(encoded)
        encoded = keras.layers.Flatten()(encoded)
        encoded = keras.layers.Dense(400)(encoded)
        encoded = PReLU()(encoded)
        encoded = keras.layers.Dense(100)(encoded)
        encoded = PReLU()(encoded)
        encoded = keras.layers.Dense(self.bottleneck, name='bneck')(encoded)
        encoded = PReLU()(encoded)
        
        decoded = keras.layers.Dense(100)(encoded)
        decoded = PReLU()(decoded)
        decoded = keras.layers.Dense(400)(decoded)
        decoded = PReLU()(decoded)
        decoded = keras.layers.Reshape((20, 20, 1))(decoded)
        decoded = keras.layers.Conv2D(5, 3, padding='same')(decoded)
        decoded = PReLU()(decoded)
        decoded = keras.layers.Conv2D(5, 3, padding='same')(decoded)
        decoded = PReLU()(decoded)
        decoded = keras.layers.UpSampling2D()(decoded)
        decoded = keras.layers.Conv2D(5, 3, padding='same')(decoded)
        decoded = PReLU()(decoded)
        decoded = keras.layers.Conv2D(10, 3, padding='same')(decoded)
        decoded = PReLU()(decoded)
        decoded = keras.layers.Conv2D(1, 3, padding='same')(decoded)
        
        self.autoencoder = keras.Model(input, decoded, name='autoencoder')
        self.autoencoder.compile(optimizer=self.optimizerAE,
                                loss='mse')


    def _genAdversarial(self):

        input = keras.layers.Input((1,))
        masses = keras.layers.BatchNormalization()(input)
        masses = keras.layers.Dense(10, activation='relu')(masses)
        masses = keras.layers.Dense(10, activation='relu')(masses)
        masses = keras.layers.Dense(12, activation='softmax', 
                                    name='adOut')(masses)

        self.adversarial = keras.Model(input, masses, name='adversarial')
        self.adversarial.compile(optimizer=self.optimizerAdversarial,
                                loss='categorical_crossentropy',
                                metrics=['acc'])

    
    # Generate the combined models
    def _genAAE(self):
        # Function to calculate the loss
        def lambdaLoss(x):
            mse = K.mean(K.square(x), axis=1, keepdims=True)
            return K.log(mse)

        
        # Sets adversarial trainable and AE untrainable or vice versa
        def adversarialTrainable(trainAdversarial):
            self.adversarial.trainable = trainAdversarial
            for layer in self.adversarial.layers:
                layer.trainable = trainAdversarial

            self.autoencoder.trainable = not trainAdversarial
            for layer in self.autoencoder.layers:
                layer.trainable = not trainAdversarial 


        input = self.autoencoder.input
        decoded = self.autoencoder(input)
        loss = keras.layers.Subtract()([input, decoded])
        loss = keras.layers.Flatten()(loss)
        loss = keras.layers.Lambda(lambdaLoss)(loss)
        masses = self.adversarial(loss)
        
        # Compile model training the AE on combined loss
        self.AAE = keras.Model(input, [decoded, masses], name='AAE')
        adversarialTrainable(trainAdversarial=False)
        self.AAE.compile(optimizer=self.optimizerAE,
                        loss=['mse', 'categorical_crossentropy'],
                        loss_weights=[1, -self.weightfactor])

        # Compile model training the adversarial
        self.AAE_adv = keras.Model(input, masses, name='AAE')
        adversarialTrainable(trainAdversarial=True)
        self.AAE_adv.compile(optimizer=self.optimizerAdversarial,
                            loss='categorical_crossentropy',
                            metrics=['acc'])


    # Main training loop for the adversarial training
    def alternatingTraining(self, images, masses, epochs=10, batchsize=256, 
                            savefreq=50, red_freq=10, reduction=0.5):
        # Gives the batch for next update
        def getBatch():
            idx = sorted(indices[batch * batchsize : (batch + 1) * batchsize])
            batch_images = images[idx]
            batch_masses = masses[idx]
            return batch_images, batch_masses


        # Save model and training histories
        def checkpoint():
            if epoch % savefreq == 0 or epoch == epochs:
                self.AAE.save('{}ep_{}.h5'.format(
                            epoch, self.timestamp))

            if epoch == epochs:
                np.savez('training_history_{}ep_{}'.format(
                            epoch, self.timestamp),
                        aae=aae_training, adversarial=adv_training)

            if epoch % red_freq == 0:
                lrCrit = K.eval(self.optimizerAdversarial.lr)
                lrAE =  K.eval(self.optimizerAE.lr)
                K.set_value(self.optimizerAE.lr, 
                            lrAE * reduction)
                K.set_value(self.optimizerAdversarial.lr,
                            lrCrit * reduction)
                print("Changed LR to: {}".format(lrCrit * reduction))


        indices = np.arange(0, len(images))
        batches = len(images) // batchsize
        
        # Arrays to store training histories
        aae_training = []
        adv_training = []
        # Loop over the epochs 
        for epoch in range(1, epochs + 1):
            print('\nEpoch: {}\t'.format(epoch), end='')
            start = time.time()
            np.random.shuffle(indices)
            
            # Loop over the batches
            for batch in range(batches):
                batch_images, batch_masses = getBatch()

                aae_training.append(self.AAE.train_on_batch(
                                    batch_images, [batch_images, batch_masses]))

                for i in range(3):
                    adv_training.append(self.AAE_adv.train_on_batch(
                                    batch_images, batch_masses))

            checkpoint()
            print('{} s'.format(int(time.time() - start)))
            print(np.mean(aae_training[-batches:], axis=0))
            print(np.mean(adv_training[-3*batches:], axis=0))


class LoLaClassifier():
    def __init__(self, nConstituents, nAdded, tag=0):
        self.tag = tag
        self.model = self._genNetwork(nConstituents, nAdded)
        pass

    def _genNetwork(self, nConstituents, nAdded):
        input = Input((4, nConstituents))
        layer = input

        # Combination layer adds nAdded linear combinations of vectors
        layer = CoLa(nAdded=nAdded, name='cola')(layer)
        # LoLa replaces the 4 vectors by physically more meaningful vectors
        layer = LoLa(name='lola')(layer) 

        # Connect to a fully connected network for classification
        layer = Flatten()(layer)
        layer = Dense(100, activation='relu')(layer)
        layer = Dense(50, activation='relu')(layer)
        layer = Dense(10, activation='relu')(layer)
        layer = Dense(2, activation='softmax')(layer)
        
        model = keras.Model(input, layer)
        return model


class EdgeConvClassifier():
    def __init__(self, inputshape, basefile='./'):
        self.model = self._genNetwork(inputshape)
        self.basefile = basefile

    def _genNetwork(self, input_shape):
        input = Input(input_shape)
        z = input
        edge1 = EdgeConv(16, n_channel_out=[64, 64, 64], nearest_ind=[0, 2])(z)
        edge2 = EdgeConv(16, n_channel_out=[128, 128, 128])(edge1)
        edge3 = EdgeConv(16, n_channel_out=[256, 256, 256])(edge2)
        z = Concatenate()([edge1, edge2, edge3, z])
        z = GlobalAveragePooling1D()(z)
        z = Dropout(0.1)(z)
        z = Dense(256, activation='relu')(z)
        z = Dropout(0.1)(z)
        z = Dense(128, activation='relu')(z)
        z = Dense(2, activation='softmax', name='test')(z)

        model = keras.Model(input, z)
        opt = keras.optimizers.Adam(lr=0.001)
        model.compile(optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['acc'])

        return model

    def train(self, momenta, labels, batchsize=512, epochs=10,
        savetag='test', callbacks=[], validation_split=0.1):

        fit = self.model.fit(momenta, labels, 
                epochs=epochs,
                batch_size=batchsize,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=2)

        self.model.save(self.basefile+'{}.h5'.format(savetag))
        np.savez(self.basefile + savetag,
            loss=fit.history['loss'],
            val_loss=fit.history['val_loss'],
            acc=fit.history['acc'],
            val_acc=fit.history['val_acc'])


if __name__ == "__main__":
    pass
