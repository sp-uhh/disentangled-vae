import sys
sys.path.append('.')

import numpy as np
import soundfile as sf
import os
from tqdm import tqdm
import math
import h5py as h5
import tempfile
import subprocess as sp
import skvideo.io
import concurrent.futures # for multiprocessing
import time
import torch
import torchaudio
from shutil import copyfile

from packages.processing.stft import stft_pytorch
from packages.processing.video import preprocess_ntcd_matlab
from packages.processing.target import clean_speech_VAD, clean_speech_IBM,\
                                noise_robust_clean_speech_IBM # because clean audio is very noisy

# Parameters
dataset_name = 'ntcd_timit'
if dataset_name == 'ntcd_timit':
    from packages.dataset.ntcd_timit import video_list, speech_list, noisy_clean_pair_dict, noisy_speech_dict

# Parameters
## Dataset
dataset_types = ['train', 'validation']
# dataset_types = ['test']

# dataset_size = 'subset'
dataset_size = 'complete'

# Labels
# labels = 'vad_labels'
labels = 'ibm_labels'

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

# Data directories
input_video_dir = os.path.join('data', dataset_size, 'raw/')
output_video_dir = os.path.join('data', dataset_size, 'processed/')

def process_write_label(args):
    # Separate args
    input_clean_file_path, output_clean_file_path, mat_file_path = args[0], args[1], args[2]

    output_wav_file = output_video_dir + output_clean_file_path
    os.makedirs(os.path.dirname(output_wav_file), exist_ok=True)

    copyfile(input_video_dir + input_clean_file_path, output_wav_file)

    # Read clean speech
    speech, fs_speech = torchaudio.load(input_video_dir + input_clean_file_path)
    speech = speech[0] # 1channel

    if fs != fs_speech:
        raise ValueError('Unexpected sampling rate')

    # Normalize audio
    speech = speech / (torch.max(torch.abs(speech)))

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

        # speech_ibm = noise_robust_clean_speech_IBM(s_t_torch.numpy(),
        #                                            s_tf_torch,
        #                                            fs=fs,
        #                                            wlen_sec=wlen_sec,
        #                                            hop_percent=hop_percent,
        #                                            center=center,
        #                                            pad_mode=pad_mode,
        #                                            pad_at_end=pad_at_end,
        #                                            vad_threshold=vad_threshold,
        #                                            eps=eps,
        #                                            ibm_threshold=ibm_threshold)

        label = speech_ibm


    # Read preprocessed video
    h5_file_path = output_video_dir + mat_file_path
    h5_file_path = os.path.splitext(h5_file_path)[0] + '_upsampled.h5'
    # h5_file_path = os.path.splitext(h5_file_path)[0] + '_normvideo.h5'
    with h5.File(h5_file_path, 'r') as file:
        video = np.array(file["X"][:])

    # Reduce frames of video or label
    if label.shape[-1] < video.shape[-1]:
        video = video[...,:label.shape[-1]]
    if label.shape[-1] > video.shape[-1]:
        label = label[...,:video.shape[-1]]

    # Store data and label in h5_file
    output_h5_file = output_video_dir + output_clean_file_path
    # output_h5_file = os.path.splitext(output_h5_file)[0] + '_' + labels + '.h5'
    output_h5_file = os.path.splitext(output_h5_file)[0] + '_' + labels + '_upsampled.h5'

    os.makedirs(os.path.dirname(output_h5_file), exist_ok=True)

    # Remove file if already exists
    if os.path.exists(output_h5_file):
        os.remove(output_h5_file)

    with h5.File(output_h5_file, 'w', rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots) as f:    
        
        # Exact shape of dataset is unknown in advance unfortunately
        # Faster writing if you know the shape in advance
        # Size of chunks corresponds to one spectrogram frame
        f.create_dataset('Y', shape=Y_shape, dtype='float32', maxshape=Y_maxshape, chunks=None, compression=compression)

        # Store dataset in variables for faster I/O
        fy = f['Y']

        # Store spectrogram in label
        fy.resize((fy.shape[-1] + label.shape[-1]), axis = len(fy.shape)-1)
        fy[...,-label.shape[-1]:] = label
    
    # Compute mean, std
    if dataset_type == 'train':
        spectrogram = np.power(abs(speech_tf), 2)

        # Apply log
        spectrogram = np.log(spectrogram + eps)

        # Reduce frames of spectrogram or label
        if label.shape[-1] < spectrogram.shape[-1]:
            spectrogram = spectrogram[...,:label.shape[-1]]
        if label.shape[-1] > spectrogram.shape[-1]:
            label = label[...,:spectrogram.shape[-1]]

        # VAR = E[X**2] - E[X]**2
        n_samples = spectrogram.shape[-1]
        channels_sum = np.sum(spectrogram, axis=-1)
        channels_squared_sum = np.sum(spectrogram**2, axis=-1)

        return n_samples, channels_sum, channels_squared_sum

def process_write_noisy_audio(args):
    # Separate args
    #TODO: modify
    noisy_file_path, clean_file_path = args[0], args[1]

    # Copy noisy files to processed
    ouput_noisy_file_path = noisy_input_output_pair_paths[noisy_file_path]
    ouput_noisy_file_path = output_video_dir + ouput_noisy_file_path
    
    os.makedirs(os.path.dirname(ouput_noisy_file_path), exist_ok=True)

    copyfile(input_video_dir + noisy_file_path, ouput_noisy_file_path)

    # Read clean speech
    noisy_speech, fs_noisy_speech = torchaudio.load(input_video_dir + noisy_file_path)
    noisy_speech = noisy_speech[0] # 1channel

    if fs != fs_noisy_speech:
        raise ValueError('Unexpected sampling rate')

    # Normalize audio
    noisy_speech = noisy_speech / (torch.max(torch.abs(noisy_speech)))

    # TF representation (PyTorch)
    noisy_speech_tf = stft_pytorch(noisy_speech,
            fs=fs,
            wlen_sec=wlen_sec,
            win=win, 
            hop_percent=hop_percent,
            center=center,
            pad_mode=pad_mode,
            pad_at_end=pad_at_end) # shape = (freq_bins, frames)

    # # Power spectrogram
    # Abs spectrogram
    noisy_speech_tf = noisy_speech_tf.numpy()
    noisy_spectrogram = noisy_speech_tf[...,0]**2 + noisy_speech_tf[...,1]**2
    #TODO: re-run with corrected magnitude spec
    # noisy_spectrogram = abs(noisy_speech_tf[...,0]) + abs(noisy_speech_tf[...,1])

    # Apply log
    noisy_spectrogram = np.log(noisy_spectrogram + eps)
    
    # Read preprocessed video corresponding to clean audio
    mat_file_path = clean_file_path.replace('Clean', 'matlab_raw')
    h5_file_path = output_video_dir + mat_file_path
    h5_file_path = os.path.splitext(h5_file_path)[0] + '_upsampled.h5'
    # h5_file_path = os.path.splitext(h5_file_path)[0] + '_normvideo.h5'
    with h5.File(h5_file_path, 'r') as file:
        video = np.array(file["X"][:])

    # Reduce frames of video or label
    if noisy_spectrogram.shape[-1] < video.shape[-1]:
        video = video[...,:noisy_spectrogram.shape[-1]]
    if noisy_spectrogram.shape[-1] > video.shape[-1]:
        noisy_spectrogram = noisy_spectrogram[...,:video.shape[-1]]

    # Compute mean, std
    if dataset_type == 'train':
        # VAR = E[X**2] - E[X]**2
        n_samples = noisy_spectrogram.shape[-1]
        channels_sum = np.sum(noisy_spectrogram, axis=-1)
        channels_squared_sum = np.sum(noisy_spectrogram**2, axis=-1)

        return n_samples, channels_sum, channels_squared_sum

def main():
    
    global dataset_type
    global noisy_input_output_pair_paths

    for dataset_type in dataset_types:

        # Create file list
        mat_file_paths = video_list(input_video_dir=input_video_dir,
                        dataset_type=dataset_type)

        input_clean_file_paths, \
            output_clean_file_paths = speech_list(input_speech_dir=input_video_dir,
                                dataset_type=dataset_type)

        args = [[input_clean_file_path, output_clean_file_path, mat_file_path]
                        for input_clean_file_path, output_clean_file_path, mat_file_path\
                            in zip(input_clean_file_paths, output_clean_file_paths, mat_file_paths)]

        t1 = time.perf_counter()

        # # VAR = E[X**2] - E[X]**2
        # n_samples, channels_sum, channels_squared_sum = 0., 0., 0.

        # for i, arg in tqdm(enumerate(args)):
        #     n_s, c_s, c_s_s = process_write_label(arg)
        #     n_samples += n_s
        #     channels_sum += c_s
        #     channels_squared_sum += c_s_s

        # # Save data on SSD....
        # with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        #     train_stats = executor.map(process_write_label, args)

        # # Process targets
        # for i, arg in tqdm(enumerate(args)):
        #     process_write_label(arg)

        with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
            # train_stats = executor.map(process_write_label, args)
            executor.map(process_write_label, args)

        t2 = time.perf_counter()
        print(f'Finished in {t2 - t1} seconds')

        # Dict mapping noisy speech to clean speech
        noisy_clean_pair_paths = noisy_clean_pair_dict(input_speech_dir=input_video_dir,
                                            dataset_type=dataset_type,
                                            dataset_size=dataset_size)

        # Dict mapping input noisy speech to output noisy speech
        noisy_input_output_pair_paths = noisy_speech_dict(input_speech_dir=input_video_dir,
                                            dataset_type=dataset_type,
                                            dataset_size=dataset_size)

        # loop over inputs for the statistics
        args = list(noisy_clean_pair_paths.items())

        # Compute mean, std of the train set
        if dataset_type == 'train':
            # # VAR = E[X**2] - E[X]**2
            n_samples, channels_sum, channels_squared_sum = 0., 0., 0.

            t1 = time.perf_counter()

            # for i, arg in tqdm(enumerate(args)):
            #     n_s, c_s, c_s_s = process_write_noisy_audio(arg)
            #     n_samples += n_s
            #     channels_sum += c_s
            #     channels_squared_sum += c_s_s

            # Save data on SSD....
            with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
                train_stats = executor.map(process_write_noisy_audio, args)
            
            for (n_s, c_s, c_s_s) in train_stats:
                n_samples += n_s
                channels_sum += c_s
                channels_squared_sum += c_s_s 

            t2 = time.perf_counter()
            print(f'Finished in {t2 - t1} seconds')

            print('Compute mean and std')
            #NB: compute the empirical std (!= regular std)
            mean = channels_sum / n_samples
            std = np.sqrt((1/(n_samples - 1))*(channels_squared_sum - n_samples * mean**2))

            # Save statistics
            # output_dataset_file = output_video_dir + os.path.join(dataset_name, 'Noisy', dataset_name + '_' + 'power_spec' + '_statistics.h5')
            # output_dataset_file = output_video_dir + os.path.join(dataset_name, 'Noisy', dataset_name + '_' + 'power_spec_upsampled' + '_statistics.h5')
            # output_dataset_file = output_video_dir + os.path.join(dataset_name, 'Noisy', dataset_name + '_' + 'log_power_spec' + '_statistics.h5')
            output_dataset_file = output_video_dir + os.path.join(dataset_name, 'Noisy', dataset_name + '_' + 'log_power_spec_upsampled' + '_statistics.h5')
            # output_dataset_file = output_video_dir + os.path.join(dataset_name, 'Clean', dataset_name + '_' + 'power_spec' + '_statistics.h5')
            # output_dataset_file = output_video_dir + os.path.join(dataset_name, 'Clean', dataset_name + '_' + 'log_power_spec' + '_statistics.h5')
            # output_dataset_file = output_video_dir + os.path.join(dataset_name, 'Clean', dataset_name + '_' + 'log_power_spec_upsampled' + '_statistics.h5')
            
            os.makedirs(os.path.dirname(output_dataset_file), exist_ok=True)

            with h5.File(output_dataset_file, 'w', rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots) as f:
                # Delete datasets if already exists
                if 'X_' + dataset_type + '_mean' in f:
                    del f['X_' + dataset_type + '_mean']
                    del f['X_' + dataset_type + '_std']

                f.create_dataset('X_' + dataset_type + '_mean', shape=X_chunks, dtype='float32', maxshape=X_chunks, chunks=None, compression=compression)
                f.create_dataset('X_' + dataset_type + '_std', shape=X_chunks, dtype='float32', maxshape=X_chunks, chunks=None, compression=compression)
                
                f['X_' + dataset_type + '_mean'][:] = mean[..., None] # Add axis to fit chunks shape
                f['X_' + dataset_type + '_std'][:] = std[..., None] # Add axis to fit chunks shape
                print('Mean and std saved in HDF5.')
        
        if dataset_type in ['validation', 'test']:

            t1 = time.perf_counter()

            # for i, arg in tqdm(enumerate(args)):
            #     process_write_noisy_audio(arg)

            with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
                executor.map(process_write_noisy_audio, args)

            t2 = time.perf_counter()
            print(f'Finished in {t2 - t1} seconds')

if __name__ == '__main__':
    main()