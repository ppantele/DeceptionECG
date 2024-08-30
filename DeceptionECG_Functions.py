import subprocess
subprocess.run(["pip", "install", "deepfake-ecg", "-q"])

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

  """
  Generate synthetic, random ECG signals and preprocess them into a standardized format.

  Parameters:
  - n (int): The number of synthetic ECG signals to generate.

  Returns:
  - signal_matrix (np.ndarray): A NumPy array of shape (n, 12, 1000) containing the 
  standardized 12-lead ECG signals. Each signal has 12 channels (leads) and 1000 data points.
  """
  
  signal_matrix = np.zeros((n, 12, 1000))
  deepfakeecg.generate(n, ".", start_id=0, run_device="cpu")
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




## HELPERS

# Transform 12 to 8, and 8 to 12 -lead signals
# Assume the 12-lead order is: 'I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
# Assume the 8-lead order is: 'I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'

def transform_12_to_8(signals_12):
  selected_channels = [0, 1, 6, 7, 8, 9, 10, 11]
  signals_8 = signals_12[:, selected_channels, :]
  return signals_8

def transform_8_to_12(signals_8):
  signals_12 = np.zeros((signals_8.shape[0], 12, signals_8.shape[2]))
  I = signals_8[:,0,:]
  II = signals_8[:,1,:]
  V1 = signals_8[:,2,:]
  V2 = signals_8[:,3,:]
  V3 = signals_8[:,4,:]
  V4 = signals_8[:,5,:]
  V5 = signals_8[:,6,:]
  V6 = signals_8[:,7,:]
  III = II - I
  aVR = -0.5*(I + II)
  aVL = I - 0.5 * II
  aVF = II - 0.5 * I
  channels = [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
  for channel in range(len(channels)):
    signals_12[:,channel,:] = channels[channel]
  return signals_12





def DeceptionECG_DiseaseSpec(input_ecg, disease):
    """
    Modify an input normal ECG to resemble a diseased ECG.
    
    Parameters:
    - input_ecg: np.ndarray
      The input normal ECG signal, shaped (n_samples, 8, 1000).
    - disease: str
      The disease to inject. Options: 'AMI', 'AFIB', 'WPW'.
    
    Returns:
    - output_ecg: np.ndarray
      The output ECG with the injected condition, shaped (n_samples, 8, 1000).
    """
    
    # Validate the disease input
    if disease not in ['AMI', 'AFIB', 'WPW']:
        raise ValueError(f"Invalid disease type '{disease}'. Choose from 'AMI', 'AFIB', 'WPW'.")
    
    # Encode the disease as a one-hot vector (assuming the model takes one-hot encoding for diseases)
    disease_dict = {'AMI': [1, 0, 0], 'AFIB': [0, 1, 0], 'WPW': [0, 0, 1]}
    disease_vector = np.array(disease_dict[disease])

    input_ecg = transform_12_to_8(input_ecg)
    input_ecg = np.transpose(input_ecg, (0, 2, 1))
  
    # Expand disease_vector to match the batch size of the input_ecg
    disease_vector = np.tile(disease_vector, (input_ecg.shape[0], 1))
    
    # Use the pre-trained model "Deception_dis_inj" to inject the disease
    output_ecg = Deception_dis_inj.predict([input_ecg, disease_vector])

    output_ecg = transform_8_to_12(output_ecg)
    output_ecg = np.transpose(output_ecg, (0, 2, 1))
    
    return output_ecg






























def DeceptionECG_SamplePlot(input, signal_n, fs=100, channel_names=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']):
  signal_n -= 1  # So that the first input signal (1st, signal_n: 1), becomes: 0 index 
  channels_n = input.shape[1] 
  for i in range(channels_n):
    plt.figure(figsize=(10, 1))
    plt.plot(input[signal_n, i, :])
    if channel_names is not None:
      text = 'Lead: ' + channel_names[i]
      box_props = dict(boxstyle='round', facecolor='white', alpha=0.5)
      plt.text(0.9, 0.9, text, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', bbox=box_props)
    plt.show()
  
  return None



def DeceptionECG_LeadPlot(input, channel=2, n_signal=10, fs=100):
  channel -= 1 # So that channel 2, becomes 1 and corresponds to lead II (0, 1, 2) (similary 1 >> I, 12 >> V6)
  channel_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
  channel_name = channel_names[channel]
  for i in range(n_signal):
    plt.figure(figsize=(10, 1))
    plt.plot(input[i, channel, :])
    text = 'Sample: ' + str(i+1) + ' (Lead: ' + channel_name + ')'
    box_props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0.9, 0.9, text, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', bbox=box_props)
    plt.show()
  return None



def DeceptionECG_SamplePlot_fine(input, signal_n=1, save=0, dpi=300):
  signal_n -= 1  # to account for 1st signal, being the 0 index
  ecg_data = input[signal_n]
  leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
  num_leads = ecg_data.shape[0]
  num_samples = ecg_data.shape[1]
  sampling_rate = 100  # 100 Hz
  duration = 10  # 10 seconds
  time_axis = np.linspace(0, duration, num_samples)

  # Create subplots for each lead
  lead_height = 1.15 # Equal to the width of 1 sec = 5 * 200ms in the x-axis
  total_height = num_leads * lead_height
  fig, axes = plt.subplots(num_leads, 1, figsize=(10, total_height), sharex=True)

  for lead, ax in enumerate(axes):
    lead_data = ecg_data[lead]
    ax.plot(time_axis, lead_data, color='black', linewidth=1)
    ax.set_ylabel(leads[lead], rotation=0, labelpad=20, fontsize=12)
    ax.set_xlim([0, duration])

    # Dynamically adjust the y-limits based on the data range with a bit of padding
    data_min = lead_data.min()
    data_max = lead_data.max()
    y_padding = 0.1 * (data_max - data_min)
    ax.set_ylim([data_min - y_padding, data_max + y_padding])

    # Calculate y-axis grid spacing to maintain squares
    time_square_size = 0.04  # 40ms (0.04s) as the square side length in time
    amplitude_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    amplitude_square_size = amplitude_range * time_square_size / duration

    lead_range = [data_min - y_padding, data_max + y_padding]

    # Vertical red lines (every 40ms for minor, every 200ms for major)
    for j in range(0, num_samples, int(sampling_rate * time_square_size)):  # Minor lines every 40ms
      ax.plot([time_axis[j], time_axis[j]], lead_range, color='red', alpha=0.05)

    for j in range(0, num_samples, int(sampling_rate * (time_square_size * 5))):  # Major lines every 200ms
      ax.plot([time_axis[j], time_axis[j]], lead_range, color='red', alpha=0.2)

    # Horizontal red lines based on the dynamic range
    for k in np.arange(lead_range[0], lead_range[1], (lead_range[1]-lead_range[0])/(5*5)):
      ax.hlines(y=k, xmin=0, xmax=duration, color='red', alpha=0.05)

    for k in np.arange(lead_range[0], lead_range[1], (lead_range[1]-lead_range[0])/5):
      ax.hlines(y=k, xmin=0, xmax=duration, color='red', alpha=0.2)
    
    # Remove unnecessary spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Hide y-axis ticks and labels
    ax.set_yticks([])
    ax.set_yticklabels([])

  axes[-1].set_xlabel('Time (s)', fontsize=12)
  axes[-1].set_xticks(np.arange(0, 11, 1))  # Set ticks at each whole second
  axes[-1].set_xticklabels([str(int(x)) for x in np.arange(0, 11, 1)])  # Set labels for each tick

  # Ensure the aspect ratio is equal to maintain square grids
  for ax in axes:
    ax.set_aspect('auto')  # Adjust the aspect ratio as needed

  # Adjust layout to prevent overlap
  plt.tight_layout()

  # Save the figure if save is 1
  if save:
    plt.savefig('ECG_generated.png', dpi=dpi, bbox_inches='tight')
    print(f'Figure saved as "ECG_generated.png" with resolution {dpi} DPI.')

  # Show the plot
  plt.show()
  return None
