import sys
sys.path.append('.')

import numpy as np
import soundfile as sf
import os
import concurrent.futures # for multiprocessing
import time
import pickle

# Parameters
## Dataset
speech_dataset_name = 'ntcd_timit'
noise_dataset_name = 'qutnoise_databases'

if speech_dataset_name == 'ntcd_timit':
    from packages.dataset.ntcd_timit import video_list, speech_list
if noise_dataset_name == 'qutnoise_databases':
    from packages.dataset.qut_database import noise_list, preprocess_noise, noise_segment, noise_list_preprocessed

dataset_type = 'test'

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
input_speech_dir = os.path.join('data', dataset_size, 'raw/')
input_noise_dir = os.path.join('data/complete/raw', noise_dataset_name + '/') # change the name of the subfolder in your computer
output_noise_dir = os.path.join('data/complete/processed', noise_dataset_name + '/') # change the name of the subfolder in your computer
output_wav_dir = os.path.join('data', dataset_size, 'processed', speech_dataset_name, noise_dataset_name + '/')

def process_noise():

    noise_types = ['cafe', 'home', 'street', 'car']

    # Create noise audios
    noise_paths = noise_list(input_noise_dir=input_noise_dir,
                             dataset_type=dataset_type)
    noise_audios = {}

    for noise_type, noise_path in noise_paths.items():
        if dataset_type == 'test':
            output_noise_path = output_noise_dir + 'si_et_05' + '/' + noise_type + '.wav'
        
        #if noise already preprocessed, read files directly
        if os.path.exists(output_noise_path):
            
            noise_audio, fs_noise = sf.read(output_noise_path)
            
            if fs != fs_noise:
                raise ValueError('Unexpected sampling rate')
        else:
            noise_audio, fs_noise = sf.read(input_noise_dir + noise_path)
            
            # Preprocess noise   
            noise_audio = preprocess_noise(noise_audio, noise_type, fs_noise, fs)

            os.makedirs(os.path.dirname(output_noise_path), exist_ok=True)
            sf.write(output_noise_path, noise_audio, fs)
        
        noise_audios[noise_type] = noise_audio

def process_save_utt(args):
    # Separate args
    input_clean_file_path, output_clean_file_path, noise_type, snr_dB  = args[0], args[1], args[2], args[3]

    speech, fs_speech = sf.read(input_speech_dir + input_clean_file_path)
    speech = speech[0] # 1channel

    # Cut burst at begining of file
    speech = speech[int(0.1*fs):]

    # Normalize audio
    speech = speech/(np.max(abs(speech)))

    if fs != fs_speech:
        raise ValueError('Unexpected sampling rate')

    # Extract noise segment
    noise = noise_segment(noise_audios, noise_type, speech)

    # Compute noise gain
    speech_power = np.sum(np.power(speech, 2))
    noise_power = np.sum(np.power(noise, 2))
    noise_power_target = speech_power*np.power(10,-snr_dB/10)
    k = noise_power_target / noise_power
    noise = noise * np.sqrt(k)

    # Normalize by max of speech, noise, speech+noise
    norm = np.max(abs(np.concatenate([speech, noise, speech+noise])))
    mixture = (speech+noise) / norm
    speech /= norm
    noise /= norm

    # Save .wav files
    output_path = output_wav_dir + output_clean_file_path
    output_path = os.path.splitext(output_path)[0]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    sf.write(output_path + '_s.wav', speech, fs)
    sf.write(output_path + '_n.wav', noise, fs)
    sf.write(output_path + '_x.wav', mixture, fs)

    # TODO: save SNR, level_s, level_n in a figure

def main():

    # Create file list
    input_clean_file_paths, \
        output_clean_file_paths = speech_list(input_speech_dir=input_speech_dir,
                            dataset_type=dataset_type)

    # Create SNR list
    np.random.seed(0)
    noise_types = ['cafe', 'home', 'street', 'car']
    #TODO: more noise index
    noise_index = np.random.randint(len(noise_types), size=len(input_clean_file_paths)) 
    # snrs = [-5.0, 0.0, 5.0]
    snrs = [-15.0, -10.0, -5.0, 0.0, 5.0]
    snrs_index = np.random.randint(len(snrs), size=len(input_clean_file_paths))

    # Create noise_audios from processed noise files
    preprocessed_noise_paths = noise_list_preprocessed(preprocessed_noise_dir=output_noise_dir,
                            dataset_type=dataset_type)
    global noise_audios # in order to be read by process_save_utt
    noise_audios = {}

    # Load the noise files
    for noise_type, preprocessed_noise_path in preprocessed_noise_paths.items():
        
        #if noise already preprocessed, read files directly
        if os.path.exists(preprocessed_noise_path):
            
            noise_audio, fs_noise = sf.read(preprocessed_noise_path)
            noise_audio = noise_audio[0] #1channel

            if fs != fs_noise:
                raise ValueError('Unexpected sampling rate')
            
            noise_audios[noise_type] = noise_audio

    # Save all SNRs
    all_snr_dB = [snrs[snrs_index[i]] for i in range(len(input_clean_file_paths))]
    # TODO: save SNR, level_s, level_n in 1 big csv
    
    os.makedirs(output_wav_dir, exist_ok=True)
    output_data_path = output_wav_dir + dataset_type + '_snr_db' + '.p'

    with open(output_data_path, 'wb') as file:
        pickle.dump(all_snr_dB, file, protocol=4)
    
    # TODO: save histogram of SNR
    # Select noise_type            
    all_noise_type = [noise_types[noise_index[i]] for i in range(len(input_clean_file_paths))]
    
    # Fuse lists
    args = [[input_clean_file_path, output_clean_file_path, noise_type, snr_dB]\
                for input_clean_file_path, output_clean_file_path, noise_type, snr_dB\
                    in zip(input_clean_file_paths, output_clean_file_paths, all_noise_type, all_snr_dB)]

    t1 = time.perf_counter()

    # with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
    #     executor.map(process_save_utt, args)
    
    # Test script on 1 sublist
    process_save_utt(args[0])

    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')
       

    #open_file(output_pickle_dir)

if __name__ == '__main__':
    # process_noise()
    main()