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
from scipy.fftpack import idct
import cv2

from packages.processing.stft import stft
from packages.processing.video import preprocess_ntcd_matlab
from packages.processing.target import clean_speech_VAD, clean_speech_IBM,\
                                noise_robust_clean_speech_IBM # because clean audio is very noisy

# Parameters
dataset_name = 'ntcd_timit'
if dataset_name == 'ntcd_timit':
    from packages.dataset.ntcd_timit import video_list, kaldi_list, speech_list

## Dataset
# dataset_types = ['train', 'validation']
dataset_types = ['test']

dataset_size = 'subset'
# dataset_size = 'complete'

# Labels
labels = 'vad_labels'
# labels = 'ibm_labels'

# ## Video
# visual_frame_rate_i = 30 # initial visual frames per second
# width = 67
# height = 67
# crf = 0 #set the constant rate factor to 0, which is lossless

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

## Video
visual_frame_rate_i = 30 # initial visual frames per second
visual_frame_rate_o = 1 / (wlen_sec * hop_percent)
width = 67
height = 67
crf = 0 #set the constant rate factor to 0, which is lossless

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

# X_shape = (height, width, 3, 0)
# X_maxshape = (height, width, 3, None)
# X_chunks = (height, width, 3, 1)

X_shape = (height, width, 0)
X_maxshape = (height, width, None)
X_chunks = (height, width, 1)

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

def process_write_video(args):
    # Separate args
    input_clean_file_path, output_clean_file_path, mat_file_path = args[0], args[1], args[2]

    # Read video
    with h5.File(input_video_dir + mat_file_path, 'r') as vfile:
        for key, value in vfile.items():
            matlab_frames_list_per_user = np.array(value)
    
    # Process video
    # Create temporary file for video upsampling
    with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp:

        # initialize video writer
        out = skvideo.io.FFmpegWriter(tmp.name,
                    inputdict={'-r': str(visual_frame_rate_i),
                            '-s':'{}x{}'.format(width,height)},
                    outputdict={'-filter:v': 'fps=fps={}'.format(visual_frame_rate_o),
                    # outputdict={'-r': str(visual_frame_rate_i),
                                '-c:v': 'libx264',
                                '-crf': str(crf),
                                '-preset': 'veryslow'}
        )
        
        # Write and upsample video
        # for frame in range(matlab_frames_list_per_user.shape[0]):
        #     rgb_rotated_df = preprocess_ntcd_matlab(matlab_frames=matlab_frames_list_per_user,
        #                         frame=frame,
        #                         width=width,
        #                         height=height,
        #                         y_hat_hard=None)
        
        A = np.zeros_like(matlab_frames_list_per_user)
        A = A.reshape(-1, width, height)
        # Write and upsample video
        for frame in range(matlab_frames_list_per_user.shape[0]):
            # rgb_rotated_df = preprocess_ntcd_matlab(matlab_frames=matlab_frames_list_per_user,
            #                     frame=frame,
            #                     width=width,
            #                     height=height,
            #                     y_hat_hard=y_hat_hard)

            data_frame = matlab_frames_list_per_user[frame]  # data frame will be shortened to "df" below
            reshaped_df = data_frame.reshape(width, height)
            idct_df = idct(idct(reshaped_df).T).T
            A[frame] = idct_df

        for frame in range(A.shape[0]):
            #TODO: take the largest max - min differences among the frames
            #TODO: divide by idct
            data_frame = A[frame]
            normalized_df = (data_frame - A.min()) / (A.max(axis=(-2,-1)) - A.min(axis=(-2,-1))).max() * 255.0
            rotated_df = np.rot90(normalized_df, 3)

            # Duplicate channel to visualize video
            rgb_rotated_df = cv2.merge([rotated_df] * 3)                       
            
            out.writeFrame(rgb_rotated_df)
                
        # close out the video writer
        out.close()
        
        # Read upsampled video
        video = skvideo.io.vread(tmp.name) # (frames, height, width, channel)
        video = np.moveaxis(video, 0,-1) # (height, width, channel, frames)
        video = np.float32(video) #convert to float32

        # Use only 1 channel
        video = video[:,:,0] # (height, width, frames)
        
        # # Just use the unprocessed matlab files
        # #TODO: upsample....
        # video = matlab_frames_list_per_user.swapaxes(0,1)
        # video = video.reshape(height, width, -1)
        # # # video = np.repeat(video[:,:,None],3,axis=2)
    
    # Read clean speech
    speech, fs_speech = sf.read(input_video_dir + input_clean_file_path)

    if fs != fs_speech:
        raise ValueError('Unexpected sampling rate')

    # Normalize audio
    speech = speech / (np.max(abs(speech)))

    # TF reprepsentation
    speech_tf = stft(speech,
            fs=fs,
            wlen_sec=wlen_sec,
            win=win,
            hop_percent=hop_percent,
            center=center,
            pad_mode=pad_mode,
            pad_at_end=pad_at_end,
            dtype=dtype) # shape = (freq_bins, frames)
    
    if labels == 'vad_labels':
        # Compute vad
        speech_vad = clean_speech_VAD(speech,
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

    # Reduce frames of video or label
    if label.shape[-1] < video.shape[-1]:
        video = video[...,:label.shape[-1]]
    if label.shape[-1] > video.shape[-1]:
        label = label[...,:video.shape[-1]]

    # Store data in h5_file
    output_h5_file = output_video_dir + mat_file_path
    output_h5_file = os.path.splitext(output_h5_file)[0] + '_upsampled.h5'
    # output_h5_file = os.path.splitext(output_h5_file)[0] + '_dct.h5'
    # output_h5_file = os.path.splitext(output_h5_file)[0] + '_.h5'
    # output_h5_file = os.path.splitext(output_h5_file)[0] + '_normvideo.h5'

    os.makedirs(os.path.dirname(output_h5_file), exist_ok=True)

    # Remove file if already exists
    if os.path.exists(output_h5_file):
        os.remove(output_h5_file)

    with h5.File(output_h5_file, 'w', rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots) as f:    
        
        # Exact shape of dataset is unknown in advance unfortunately
        # Faster writing if you know the shape in advance
        # Size of chunks corresponds to one spectrogram frame
        f.create_dataset('X', shape=X_shape, dtype='float32', maxshape=X_maxshape, chunks=None, compression=compression)

        # Store dataset in variables for faster I/O
        fx = f['X']

        # Store spectrogram in dataset
        fx.resize((fx.shape[-1] + video.shape[-1]), axis = len(fx.shape)-1)
        fx[...,-video.shape[-1]:] = video

    # Store label in h5_file
    output_h5_file = output_video_dir + output_clean_file_path
    output_h5_file = os.path.splitext(output_h5_file)[0] + '_' + labels + '.h5'

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
        # VAR = E[X**2] - E[X]**2
        # n_samples = video.shape[-1]
        # channels_sum = np.sum(video, axis=-1)
        # channels_squared_sum = np.sum(video**2, axis=-1)

        n_samples = np.prod(video.shape)
        channels_sum = np.sum(video)
        channels_squared_sum = np.sum(video**2)

        # n_samples = video.shape[-1]
        # mean_pixel = np.mean(video, axis=(0,1))
        # channels_sum = np.sum(mean_pixel)
        # channels_squared_sum = np.sum(mean_pixel**2)

        return n_samples, channels_sum, channels_squared_sum

def main():
    
    global dataset_type

    for dataset_type in dataset_types:

        # Create file list
        mat_file_paths = video_list(input_video_dir=input_video_dir,
                                dataset_type=dataset_type)

        # Create file list
        ark_file_paths, scp_file_paths = kaldi_list(input_video_dir=input_video_dir,
                                dataset_type=dataset_type)

        input_clean_file_paths, \
            output_clean_file_paths = speech_list(input_speech_dir=input_video_dir,
                                dataset_type=dataset_type)

        args = [[input_clean_file_path, output_clean_file_path, mat_file_path]
                        for input_clean_file_path, output_clean_file_path, mat_file_path\
                            in zip(input_clean_file_paths, output_clean_file_paths, mat_file_paths)]

        # Compute mean, std of the train set
        if dataset_type == 'train':
            # VAR = E[X**2] - E[X]**2
            n_samples, channels_sum, channels_squared_sum = 0., 0., 0.

            t1 = time.perf_counter()

            # for i, arg in tqdm(enumerate(args)):
            #     n_s, c_s, c_s_s = process_write_video(arg)
            #     n_samples += n_s
            #     channels_sum += c_s
            #     channels_squared_sum += c_s_s

            with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
                train_stats = executor.map(process_write_video, args)
            
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
            # output_dataset_file = output_video_dir + os.path.join(dataset_name, 'matlab_raw', dataset_name + '_' + 'pixel' + '_statistics.h5')
            # output_dataset_file = output_video_dir + os.path.join(dataset_name, 'matlab_raw', dataset_name + '_' + 'pixel_dct' + '_statistics.h5')
            # output_dataset_file = output_video_dir + os.path.join(dataset_name, 'matlab_raw', dataset_name + '_' + 'dct' + '_statistics.h5')
            # output_dataset_file = output_video_dir + os.path.join(dataset_name, 'matlab_raw', dataset_name + '_statistics.h5')
            # output_dataset_file = output_video_dir + os.path.join(dataset_name, 'matlab_raw', dataset_name + '_' + 'normvideo' + '_statistics.h5')
            output_dataset_file = output_video_dir + os.path.join(dataset_name, 'matlab_raw', dataset_name + '_' + 'upsampled' + '_statistics.h5')
            if not os.path.exists(os.path.dirname(output_dataset_file)):
                os.makedirs(os.path.dirname(output_dataset_file))

            with h5.File(output_dataset_file, 'w', rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots) as f:
                # Delete datasets if already exists
                if 'X_' + dataset_type + '_mean' in f:
                    del f['X_' + dataset_type + '_mean']
                    del f['X_' + dataset_type + '_std']

                # f.create_dataset('X_' + dataset_type + '_mean', shape=X_chunks, dtype='float32', maxshape=X_chunks, chunks=None, compression=compression)
                # f.create_dataset('X_' + dataset_type + '_std', shape=X_chunks, dtype='float32', maxshape=X_chunks, chunks=None, compression=compression)
                f.create_dataset('X_' + dataset_type + '_mean', shape=(1,1), dtype='float32', maxshape=(1,1), chunks=None, compression=compression)
                f.create_dataset('X_' + dataset_type + '_std', shape=(1,1), dtype='float32', maxshape=(1,1), chunks=None, compression=compression)
                
                f['X_' + dataset_type + '_mean'][:] = mean[..., None] # Add axis to fit chunks shape
                f['X_' + dataset_type + '_std'][:] = std[..., None] # Add axis to fit chunks shape
                print('Mean and std saved in HDF5.')

        if dataset_type in ['validation', 'test']:

            t1 = time.perf_counter()

            # for i, arg in tqdm(enumerate(args)):
            #     process_write_video(arg)

            with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
                executor.map(process_write_video, args)

            t2 = time.perf_counter()
            print(f'Finished in {t2 - t1} seconds')

if __name__ == '__main__':
    main()