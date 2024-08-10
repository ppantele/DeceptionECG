import subprocess
subprocess.run(["pip", "install", "deepfake-ecg"])

import deepfakeecg
import numpy as np
import os
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import csv, random, os
from datetime import datetime



def DeceptionECG_GenPurp(n):

  deepfakeecg.generate(n, ".", start_id=0, run_device="cpu")

  signal_matrix = np.zeros((n, 12, 1000))

  
  for sample in range(n):
  
    with open(str(sample)+'.asc', 'r') as file:
      content = file.read()
    lines = content.strip().split('\n')
    data = []
    for line in lines:
      numbers = [int(x) for x in line.split()]
      data.append(numbers)
    sample_matrix = np.array(data).T[:,::5]

    I = sample_matrix[0,:]
    II = sample_matrix[1,:]
    V1 = sample_matrix[2,:]
    V2 = sample_matrix[3,:]
    V3 = sample_matrix[4,:]
    V4 = sample_matrix[5,:]
    V5 = sample_matrix[6,:]
    V6 = sample_matrix[7,:]
    III = II - I
    aVR = -0.5*(I + II)
    aVL = I - 0.5 * II
    aVF = II - 0.5 * I
    channels = [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
    for channel in range(len(channels)):
      signal_matrix[sample,channel,:] = channels[channel]

  signal_matrix = (signal_matrix - np.mean(signal_matrix)) / np.std(signal_matrix) # standardize

  for sample in range(n):
    os.remove(str(sample)+'.asc')

  return signal_matrix



















def DeceptionECG_DiseaseSpec(input, disease):
  return 1
























class Model:
  # Generic model super-class to be overriden
  def __init__(self):
    self._model = None

  def _build_model(self):
    raise NotImplementedError("Subclasses must implement the speak method")

  def fit(self, X_train, y_train):
    self._model.fit(X_train, y_train)

  def predict(self, X_test):
    return self._model.predict(X_test)

  def train_on_batch(self, X_train_batch, y_train_batch):
    return self._model.train_on_batch(X_train_batch, y_train_batch)

  def evaluate(self, X_test, y_test):
    return self._model.evaluate(X_test, y_test, verbose=0)

  def expose_model(self):
    return self._model


class Generator(Model):

  def __init__(self, ecg_seq_length, ecg_channels,
               filters = [16, 32, 128, 256],
               strides = [1, 1, 1, 1],
               kernels = [2, 10, 40],
               scaling_factor = 8,
               BN = False, RES = True, lRLa = 0.2):
    self._model = self._build_model(ecg_seq_length, ecg_channels,
                                    filters, strides, kernels,
                                    scaling_factor, BN, RES, lRLa)


  def _build_model(self, ecg_seq_length, ecg_channels,
                   filters, strides, kernels, scaling_factor, BN, RES, lRLa):

    input_layer = layers.Input(shape=(ecg_seq_length, ecg_channels))
    x = input_layer

    if RES:
      shortcuts = []

    # Conv
    for f, s in zip(filters, strides):
      if RES:
        shortcuts.append(x)
      x = self._inc_module(x, Conv0_Deconv1=0, kernels=kernels, f=f, s=s, lRLa=lRLa, BN=BN)
    # last shortcut (n_steps+1_th)
    if RES:
      shortcuts.append(x)

    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(alpha=lRLa)(x)

    # DeConv
    for i, (f, s) in enumerate(zip(list(reversed(filters)), list(reversed(strides)))):
      x = self._inc_module(x, Conv0_Deconv1=1, kernels=kernels, f=f, s=s, lRLa=lRLa, BN=BN)
      if RES:
        sc = list(reversed(shortcuts))[i]
        sc = layers.Conv1D(filters=sc.shape[2], kernel_size=1, strides=1, padding='same')(sc)
        x = layers.Add()([x, sc])

    x = layers.Conv1D(filters=ecg_channels, kernel_size=40, strides=1, padding='same',
                      activation='relu', use_bias=False)(x)
    if RES:
      x = layers.Add()([x, list(reversed(shortcuts))[-1]])
    x = layers.Activation('tanh')(x)
    x = layers.Lambda(lambda x: x * scaling_factor)(x)
    output_layer = x

    # Build the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model


  def _inc_module(self, x, Conv0_Deconv1, kernels, f, s, lRLa, BN):
    # x: nd.array(n_samples*length*channels)
    # Conv0_Deconv1: 0 for Conv, 1 for DeConv
    # kernels: list, f: int, s: int, lRLa: int, BN: bool

    conv_list = []
    for kernel in kernels:
      if Conv0_Deconv1==0:
        conv_list.append(layers.Conv1D(filters=f, kernel_size=kernel, strides=s,
                                      padding='same', activation='relu', use_bias=False)(x))
      else:
        conv_list.append(layers.Conv1DTranspose(filters=f, kernel_size=kernel, strides=s,
                                                padding='same', activation='relu', use_bias=False)(x))
      x = layers.Concatenate(axis=2)(conv_list)
    if BN:
      x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=lRLa)(x)
    return x



class Discriminator(Model):
  def __init__(self, ecg_seq_length, ecg_channels,
               lr = 0.0001, beta = 0.5,
               filters = [32, 64, 128, 256, 512],
               kernels = [2, 4, 10, 15, 20],
               strides = [1, 1, 1, 1, 1],
               BN=False, DO=True, RES=True, lRLa=0.2):
    self._model = self._build_model(ecg_seq_length, ecg_channels,
                                    lr, beta, filters, kernels, strides,
                                    BN, DO, RES, lRLa)

  # Define the discriminator
  def _build_model(self, ecg_seq_length, ecg_channels,
                   lr, beta, filters, kernels, strides,
                   BN, DO, RES, lRLa):

    input_layer = layers.Input(shape=(ecg_seq_length, ecg_channels))
    x = input_layer

    for i, (f, ks, s) in enumerate(zip(filters, kernels, strides)):
      if (i%2==1):
        shortcut = x
        shortcut = layers.MaxPool1D(pool_size=s, strides=s, padding='same')(shortcut)
      x = layers.Conv1D(filters=f, kernel_size=ks, strides=s, padding='same')(x)
      if BN:
        x = layers.BatchNormalization()(x)
      x = layers.LeakyReLU(alpha=lRLa)(x)
      if DO:
        x = layers.Dropout(0.2)(x)
      if (i%2==0 and i>1):
        if RES:
          shortcut = layers.Conv1D(filters=f, kernel_size=1, strides=1, padding='same')(shortcut)
          shortcut = layers.MaxPool1D(pool_size=s, strides=s, padding='same')(shortcut)
          x = layers.Add()([x, shortcut])
        x = layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    output_layer = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    opt = Adam(learning_rate=lr, beta_1=beta)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model



class GAN(Model):

  def __init__(self, generator, discriminator,
               lr = 0.0001, beta = 0.5):
    self._model = self._build_model(generator, discriminator, lr, beta)

  def _build_model(self, generator, discriminator, lr, beta):
    model = models.Sequential()
    discriminator.trainable = False
    model.add(generator.expose_model())
    model.add(discriminator.expose_model())
    opt = Adam(learning_rate=lr, beta_1=beta)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model






class DeceptionECG:

  def __init__(self, ecg_seq_length, ecg_channels):
    self._ecg_seq_length = ecg_seq_length
    self._ecg_channels = ecg_channels
    self._d = None
    self._g = None
    self._gan = None

  def build_discriminator(self, lr, beta, 
                          filters, kernels, strides,
                          BN, DO, RES, lRLa):
                          
    self._d = Discriminator(self._ecg_seq_length, self._ecg_channels,
                            lr = lr, beta = beta,
                            filters = filters,
                            kernels = kernels,
                            strides = strides,
                            BN=BN, DO=DO, RES=RES, lRLa=lRLa)
    

  def build_generator(self, lr, beta, 
                      filters, strides, kernels,
                      scaling_factor, BN, RES, lRLa):
    
    if self._d == None:
      print("Generator cannot be created. Discriminator must be defined earlier.")
      return None

    self._g = Generator(self._ecg_seq_length, self._ecg_channels,
                        filters = filters,
                        strides = strides,
                        kernels = kernels,
                        scaling_factor = scaling_factor, BN = BN, RES = RES, lRLa = lRLa)
    
    self._gan = GAN(self._g, self._d, lr = lr, beta = beta)

  

  # HELPERs

  # check dimensions
  def _false_dims(self, dataset):
    if (dataset.shape[1]!=self._ecg_seq_length or dataset.shape[2]!=self._ecg_channels):
      print(f"Input samples must have length: {self._ecg_seq_length} and channels: {self._ecg_channels}.")
      return True
    else:
      return False

  # select real samples
  def _generate_real_samples(self, dataset, n_samples):
    # choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = np.ones((n_samples, 1))
    return X, y

  # generate points in latent space as input for the generator
  def _generate_latent_points(self, func_latent, n_samples):
    # generate points in the latent space
    # func_latent : Function to create samples, with n_samples as passed arg
    x = func_latent(n_samples)
    if self._false_dims(x):
      return None
    else:
      return x

  # use the generator to generate n fake examples, with class labels
  def _generate_fake_samples(self, g_model, func_latent, n_samples):
    # generate points in latent space
    x_input = func_latent(n_samples)
    if self._false_dims(x_input):
      return None
    else:
      # predict outputs
      X = g_model.predict(x_input)
      # create 'fake' class labels (0)
      y = np.zeros((n_samples, 1))
      return X, y

  # create and save a plot of generated images
  def _save_plot(self, n_examples, epoch, n=5, end=500, lead=0, folder_name='files'):

    # create subplots with n rows
    fig, axes = plt.subplots(n, 1, figsize=(8, 2 * n))
    # plot each image in a separate subplot
    for i in range(n):
      # define subplot
      axes[i].plot(n_examples[i, :end, lead])
      # set subplot title if needed
      axes[i].set_title(f'Example {i+1}')
    # adjust layout to prevent clipping of titles
    plt.tight_layout()
    # save plot to file
    filename = folder_name + '/generated_plot_e%03d.png' % (epoch + 1)
    plt.savefig(filename)
    plt.close()

  # Function to save losses lists to CSV
  def _save_to_csv(self, epoch, d_loss, g_loss, folder_name='files'):
      filename = folder_name + '/generator_losses_e%03d.csv' % (epoch+1)
      with open(filename, 'w', newline='') as csvfile:
          writer = csv.writer(csvfile)
          writer.writerow(['d_loss', 'g_loss'])  # Header
          for row in zip(d_loss, g_loss):
              writer.writerow(row)

  # evaluate the discriminator, plot generated images, save generator model
  def _summarize_performance(self, epoch, dataset, func_latent, lead=0, n_samples=100, folder_name='files'):
    # prepare real samples
    X_real, y_real = self._generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = self._d.evaluate(X_real, y_real)
    # prepare fake examples
    x_fake, y_fake = self._generate_fake_samples(self._g, func_latent, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = self._d.evaluate(x_fake, y_fake)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    # save plot
    self._save_plot(x_fake, epoch, n=5, end=500, lead=lead, folder_name=folder_name)
    # save the generator model tile file
    filename = folder_name + '/generator_model_e%03d.h5' % (epoch+1)
    self._g.expose_model().save(filename)

  # plot losses
  def _plot_losses(self, d_loss_l, g_loss_l):
    plt.plot(d_loss_l, label='d')
    plt.plot(g_loss_l, label='g')
    plt.legend()
    plt.show()
  

  # Train the model

  # train function
  def train(self, epochs, batch_size, X_train, func_latent, lead_to_show=0, epoch_checkpoint=1, folder_name='files_train'):

    # Check if generator, discriminator have been compiled
    if (self._g==None or self._g==None):
      print("Generator and discriminator must be compiled before training.")
      return None

    # Check dataset dims
    if self._false_dims(X_train):
      return None
    else:
      pass

    # Get the current date and time
    current_time_1 = datetime.now()

    # Batches
    half_batch = int(batch_size / 2)
    batch_n = int(X_train.shape[0] / half_batch)
    indices = list(range(X_train.shape[0]))
    random.shuffle(indices)

    # Save progress
    d_loss_l = []
    g_loss_l = []

    # Custom training loop for the GAN
    for epoch in range(epochs):
      for batch_i in range(batch_n):

        # Define batch
        start = batch_i * half_batch
        end = min(start + half_batch, X_train.shape[0])
        batches_indices = indices[start:end]

        # X, y real and fake
        X_real = X_train[batches_indices]
        y_real = np.ones((half_batch, 1))
        X_fake, y_fake = self._generate_fake_samples(self._g, func_latent, half_batch)

        # Mix and shuffle
        X_p = np.concatenate([X_real, X_fake])
        y_p = np.concatenate([y_real, y_fake])
        final_indices = np.arange(X_p.shape[0])
        np.random.shuffle(final_indices)
        X_p = X_p[final_indices]
        y_p = y_p[final_indices]

        # Train
        d_loss, d_acc = self._d.train_on_batch(X_p, y_p)
        X_gan = self._generate_latent_points(func_latent, half_batch)
        y_gan = np.ones((half_batch, 1))
        g_loss = self._gan.train_on_batch(X_gan, y_gan)

        # summarize loss on this batch
        print('>%d:%d, d_l=%.3f, d_a=%.3f, g_l=%.3f' %
        (epoch+1, batch_i+1, d_loss, d_acc, g_loss))

      # Save losses
      d_loss_l.append(d_loss)
      g_loss_l.append(g_loss)
      os.makedirs(folder_name, exist_ok=True)
      self._save_to_csv(epoch, d_loss_l, g_loss_l, folder_name=folder_name)

      # evaluate the model performance, sometimes
      if (epoch+1) % epoch_checkpoint == 0:
        self._summarize_performance(epoch, X_train, func_latent, lead=lead_to_show, folder_name=folder_name)
      # Get the current date and time and format it directly
      current_time_2 = datetime.now()
      time_diff = current_time_2 - current_time_1
      minutes, seconds = divmod(time_diff.seconds, 60)
      print(f"Time Difference: {minutes} minutes, {seconds} seconds")
      current_time_1 = current_time_2

    return d_loss, g_loss_l, time_diff






'''
# Define GAN model and input/output dimensions, e.g.:
GAN_model = DeceptionECG(1000,12) # length: 1000, channels: 12

# Define and compile  discriminator, e.g.:
GAN_model.build_discriminator(lr=0.0001, beta=0.5,
                              filters=[32, 64, 128, 256], 
                              kernels=[2, 4, 8, 16], 
                              strides=[1, 1, 1, 1],
                              BN=True, DO=True, RES=True, lRLa=0.2)
# number of filters, kernels, strides must be equal

# Define and compile generator, e.g.:
GAN_model.build_generator(lr=0.0002, beta=0.5,
                          filters = [32, 64, 128, 256],
                          strides = [1, 1, 1, 1],   
                          kernels = [2, 10, 40],
                          scaling_factor=8, BN=False, RES=True, lRLa=0.2)
# number of filters and strides must be equal
# number of "sliding windows" of different sizes

# Train GAN model
GAN_model.train(epochs=1000, batch_size=64, 
                X_train=X_train, 
                func_latent=func_example, 
                lead_to_show=1, epoch_checkpoint=1, folder_name='train_try_1')
# where:
# X_train --> np.ndarray of shape: (n_samples, length, channels), e.g.: (100, 1000, 12)
# func_latent --> returns latent input
# def func_latent(n_samples):
#   ...
#   ...
#   return X_latent 
# X_latent of shape (n_samples, length, channels), e.g.: (100, 1000, 12)
'''
