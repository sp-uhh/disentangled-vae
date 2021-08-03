import sys
sys.path.append('.')

import numpy as np
import soundfile as sf
import os
import concurrent.futures # for multiprocessing
import time
import pickle
from tqdm import tqdm
from shutil import copyfile

# Parameters
## Dataset
speech_dataset_name = 'ntcd_timit'
noise_dataset_name = 'ntcd_timit'

if speech_dataset_name == 'ntcd_timit':
    from packages.dataset.ntcd_timit import video_list, speech_list
if noise_dataset_name == 'ntcd_timit':
    from packages.dataset.ntcd_timit import noisy_clean_pair_dict, noisy_speech_dict, speech_list

# Parameters
## Dataset
dataset_types = ['test']

# dataset_size = 'subset'
dataset_size = 'complete'

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

# Data directories
input_video_dir = os.path.join('data', dataset_size, 'raw/')
output_wav_dir = os.path.join('data', dataset_size, 'processed/')

def process_write_clean_audio(args):
    # Separate args
    input_clean_file_path, output_clean_file_path = args[0], args[1]

    output_wav_file = output_wav_dir + output_clean_file_path
    os.makedirs(os.path.dirname(output_wav_file), exist_ok=True)

    copyfile(input_video_dir + input_clean_file_path, output_wav_file)

def process_write_noisy_audio(args):
    # Separate args
    noisy_file_path, clean_file_path = args[0], args[1]

    # Copy noisy files to processed
    ouput_noisy_file_path = noisy_input_output_pair_paths[noisy_file_path]
    ouput_noisy_file_path = output_wav_dir + ouput_noisy_file_path
    
    os.makedirs(os.path.dirname(ouput_noisy_file_path), exist_ok=True)

    copyfile(input_video_dir + noisy_file_path, ouput_noisy_file_path)

def main():
    
    global dataset_type
    global noisy_input_output_pair_paths

    for dataset_type in dataset_types:

        # Clean speech
        input_clean_file_paths, \
            output_clean_file_paths = speech_list(input_speech_dir=input_video_dir,
                                dataset_type=dataset_type)

        args = [[input_clean_file_path, output_clean_file_path]
                        for input_clean_file_path, output_clean_file_path\
                            in zip(input_clean_file_paths, output_clean_file_paths)]

        t1 = time.perf_counter()
        
        # Process targets
        for i, arg in tqdm(enumerate(args)):
            process_write_clean_audio(arg)

        # with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        #     # train_stats = executor.map(process_write_label, args)
        #     executor.map(process_write_clean_audio, args)

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
        
        if dataset_type in ['test']:

            t1 = time.perf_counter()

            # for i, arg in tqdm(enumerate(args)):
            #     process_write_noisy_audio(arg)

            with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
                executor.map(process_write_noisy_audio, args)

            t2 = time.perf_counter()
            print(f'Finished in {t2 - t1} seconds')

if __name__ == '__main__':
    main()