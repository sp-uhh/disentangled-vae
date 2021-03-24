import sys
sys.path.append('.')

import numpy as np
import soundfile as sf
import os
from tqdm import tqdm
import math
import h5py as h5
import torch
import torchaudio

from packages.processing.stft import stft_pytorch
from packages.processing.video import preprocess_ntcd_matlab
from packages.processing.target import clean_speech_VAD, clean_speech_IBM,\
                                noise_robust_clean_speech_IBM # because clean audio is very noisy


# Parameters
dataset_name = 'ntcd_timit'
if dataset_name == 'ntcd_timit':
    from packages.dataset.ntcd_timit import video_list, speech_list

# Parameters
## Dataset
dataset_types = ['train', 'validation']

dataset_size = 'subset'
# dataset_size = 'complete'

# Labels
labels = 'vad_labels'
# labels = 'ibm_labels'

# ## Video
# visual_frame_rate_i = 30 # initial visual frames per second

## STFT
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
# hop_percent = math.floor((1 / (wlen_sec * visual_frame_rate_i)) * 1e4) / 1e4  # hop size as a percentage of the window length
hop_percent = 0.25 # hop size as a percentage of the window length
win = 'hann' # type of window
center = False # see https://librosa.org/doc/0.7.2/_modules/librosa/core/spectrum.html#stft
pad_mode = 'reflect' # This argument is ignored if center = False
pad_at_end = True # pad audio file at end to match same size after stft + istft
dtype = 'complex64'

## Noise robust VAD
vad_threshold = 1.70

## Noise robust IBM
eps = 1e-8
ibm_threshold = 50 # Hard threshold
# ibm_threshold = 65 # Soft threshold

# HDF5 parameters
rdcc_nbytes = 1024**2*40 # The number of bytes to use for the chunk cache
                          # Default is 1 Mb
                          # Here we are using 40Mb of chunk_cache_mem here
rdcc_nslots = 1e4 # The number of slots in the cache's hash table
                  # Default is 521
                  # ideally 100 x number of chunks that can be fit in rdcc_nbytes
                  # (see https://docs.h5py.org/en/stable/high/file.html?highlight=rdcc#chunk-cache)
                  # for compression 'zlf' --> 1e4 - 1e7
                  # for compression 32001 --> 1e4

X_shape = (513, 0)
X_maxshape = (513, None)
X_chunks = (513, 1)

if labels == 'vad_labels':
    y_dim = 1
if labels == 'ibm_labels':
    y_dim = 513

Y_shape = (y_dim, 0)
Y_maxshape = (y_dim, None)
Y_chunks = (y_dim, 1)

compression = 'lzf'
shuffle = False

# Data directories
input_video_dir = os.path.join('data', dataset_size, 'raw/')
output_video_dir = os.path.join('data', dataset_size, 'processed/')
output_dataset_file = os.path.join(output_video_dir, dataset_name, 'Clean' + '_' + labels + '_upsampled.h5')

def main():

    os.makedirs(os.path.dirname(output_dataset_file), exist_ok=True)

    with h5.File(output_dataset_file, 'a', rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots) as f:    

        for dataset_type in dataset_types:

            # Create file list
            input_clean_file_paths, \
                output_clean_file_paths = speech_list(input_speech_dir=input_video_dir,
                                    dataset_type=dataset_type)

            # Create file list
            mat_file_paths = video_list(input_video_dir=input_video_dir,
                                    dataset_type=dataset_type)

            # Delete datasets if already exists
            if 'X_' + dataset_type in f:
                del f['X_' + dataset_type]
                del f['Y_' + dataset_type]
            
            # Exact shape of dataset is unknown in advance unfortunately
            # Faster writing if you know the shape in advance
            # Size of chunks corresponds to one spectrogram frame
            f.create_dataset('X_' + dataset_type, shape=X_shape, dtype='float32', maxshape=X_maxshape, chunks=X_chunks, compression=compression, shuffle=shuffle)
            f.create_dataset('Y_' + dataset_type, shape=Y_shape, dtype='float32', maxshape=Y_maxshape, chunks=Y_chunks, compression=compression, shuffle=shuffle)
                        
            # Store dataset in variables for faster I/O
            fx = f['X_' + dataset_type]
            fy = f['Y_' + dataset_type]            

            # Compute mean, std of the train set
            if dataset_type == 'train':
                # VAR = E[X**2] - E[X]**2
                channels_sum, channels_squared_sum = 0., 0.

            for i, (input_clean_file_path, mat_file_path) \
                in tqdm(enumerate(zip(input_clean_file_paths, mat_file_paths))):

                # Read clean speech
                speech, fs_speech = torchaudio.load(input_video_dir + input_clean_file_path)
                speech = speech[0] # 1channel

                if fs != fs_speech:
                    raise ValueError('Unexpected sampling rate')

                # Normalize audio
                speech = speech/(torch.max(torch.abs(speech)))

                # TF representation (PyTorch)
                speech_tf = stft_pytorch(speech,
                        fs=fs,
                        wlen_sec=wlen_sec,
                        win=win, 
                        hop_percent=hop_percent,
                        center=center,
                        pad_mode=pad_mode,
                        pad_at_end=pad_at_end) # shape = (freq_bins, frames)
                
                # Real + j * Img
                speech_tf = speech_tf[...,0].numpy() + 1j * speech_tf[...,1].numpy()

                # Spectrogram
                spectrogram = np.power(abs(speech_tf), 2)

                if labels == 'vad_labels':
                    # Compute vad
                    speech_vad = clean_speech_VAD(speech.numpy(),
                                                fs=fs,
                                                wlen_sec=wlen_sec,
                                                hop_percent=hop_percent,
                                                center=center,
                                                pad_mode=pad_mode,
                                                pad_at_end=pad_at_end,
                                                vad_threshold=vad_threshold)

                    label = speech_vad

                if labels == 'ibm_labels':
                    # binary mask
                    speech_ibm = clean_speech_IBM(speech_tf,
                                                eps=eps,
                                                ibm_threshold=ibm_threshold)

                    label = speech_ibm
                
                # Read video
                with h5.File(input_video_dir + mat_file_path, 'r') as vfile:
                    for key, value in vfile.items():
                        video = np.array(value)

                # Reduce frames of label if video is shorter
                if label.shape[-1] > video.shape[0]:
                    label = label[...,:video.shape[0]]

                # Store spectrogram in dataset
                fx.resize((fx.shape[-1] + spectrogram.shape[-1]), axis = fx.ndim-1)
                fx[...,-spectrogram.shape[-1]:] = spectrogram

                # Store spectrogram in label
                fy.resize((fy.shape[-1] + label.shape[-1]), axis = fy.ndim-1)
                fy[...,-label.shape[-1]:] = label

                # Compute mean, std
                if dataset_type == 'train':
                    # VAR = E[X**2] - E[X]**2
                    channels_sum += np.sum(spectrogram, axis=-1)
                    channels_squared_sum += np.sum(spectrogram**2, axis=-1)
            
            if dataset_type == 'train':
                print('Compute mean and std')
                #NB: compute the empirical std (!= regular std)
                n_samples = fx.shape[-1]
                mean = channels_sum / n_samples
                std = np.sqrt((1/(n_samples - 1))*(channels_squared_sum - n_samples * mean**2))
                
                # Delete datasets if already exists
                if 'X_' + dataset_type + '_mean' in f:
                    del f['X_' + dataset_type + '_mean']
                    del f['X_' + dataset_type + '_std']

                f.create_dataset('X_' + dataset_type + '_mean', shape=X_chunks, dtype='float32', maxshape=X_chunks, chunks=None, compression=compression, shuffle=shuffle)
                f.create_dataset('X_' + dataset_type + '_std', shape=X_chunks, dtype='float32', maxshape=X_chunks, chunks=None, compression=compression, shuffle=shuffle)
                
                f['X_' + dataset_type + '_mean'][:] = mean[..., None] # Add axis to fit chunks shape
                f['X_' + dataset_type + '_std'][:] = std[..., None] # Add axis to fit chunks shape
                print('Mean and std saved in HDF5.')

    
if __name__ == '__main__':
    main()