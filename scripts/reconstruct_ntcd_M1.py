import sys
sys.path.append('.')

import os
import numpy as np
import torch
import time
import soundfile as sf
from tqdm import tqdm
import librosa
import matplotlib.pyplot as plt
import h5py as h5

from packages.processing.stft import stft, istft
from packages.processing.target import clean_speech_VAD
from packages.models.models import VariationalAutoencoder

from packages.visualization import display_multiple_signals

##################################### SETTINGS #####################################################

# Dataset
speech_dataset_name = 'ntcd_timit'
noise_dataset_name = 'ntcd_timit'

if speech_dataset_name == 'ntcd_timit':
    from packages.dataset.ntcd_timit import proc_noisy_clean_pair_dict, speech_list

dataset_type = 'test'

dataset_size = 'subset'
# dataset_size = 'complete'

# Labels
labels = 'vad_labels'
# labels = 'ibm_labels'
upsampled = True

# Parameters
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

# GPU Processing
cuda = torch.cuda.is_available()

# Hyperparameters
# M1
model_name = 'ntcd_M1_nonorm_hdim_128_128_zdim_016_end_epoch_500/M1_epoch_118_vloss_416.54'
x_dim = 513
y_dim = 1
z_dim = 16
h_dim = [128, 128]
std_norm = False
eps = 1e-8

## Classifier
# classif_type = 'dnn'
classif_type = 'oracle'

if classif_type == 'oracle':
    classif_name = 'oracle_classif'
    # classif_name = 'ones_classif'
    # classif_name = 'zeros_classif'

## Plot spectrograms
vmin = -40 # in dB
vmax = 20 # in dB
xticks_sec = 0.5 # in seconds
fontsize = 30

# Data directories
input_speech_dir = os.path.join('data',dataset_size,'raw/')
processed_data_dir = os.path.join('data',dataset_size,'processed/')
processed_wav_dir = os.path.join('data', dataset_size, 'processed/')
model_path = os.path.join('models', model_name + '.pt')
output_data_dir = os.path.join('data', dataset_size, 'models', model_name + '/')

def main():
    file = open('output.log','w') 

    print('Torch version: {}'.format(torch.__version__))
    device = torch.device("cuda" if cuda else "cpu")

    print('Device: %s' % (device))
    if torch.cuda.device_count() >= 1: print("Number GPUs: ", torch.cuda.device_count())

    print('Load models')
    model = VariationalAutoencoder([x_dim, z_dim, h_dim])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    # model = model.enc_dec_clf # Exclude auxiliary from the model
    
    if cuda: model = model.cuda()

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Dict mapping noisy speech to clean speech
    noisy_clean_pair_paths = proc_noisy_clean_pair_dict(input_speech_dir=processed_data_dir,
                                            dataset_type=dataset_type,
                                            dataset_size=dataset_size,
                                            labels=labels,
                                            upsampled=upsampled)

    # Convert dict to tuples
    noisy_clean_pair_paths = list(noisy_clean_pair_paths.items())

    print('- Number of test samples: {}'.format(len(noisy_clean_pair_paths)))

    for i, (proc_noisy_file_path, clean_file_path) in tqdm(enumerate(noisy_clean_pair_paths)):
        
        # Extract input SNR
        snr_db = int(proc_noisy_file_path.split('/')[3])

        # Read video
        h5_file_path = clean_file_path.replace('Clean', 'matlab_raw')
        h5_file_path = h5_file_path.replace('_' + labels, '')
        # h5_file_path = os.path.splitext(h5_file_path)[0] + '_upsampled.h5'
        # h5_file_path = os.path.splitext(h5_file_path)[0] + '_normvideo.h5'
        h5_file_path = processed_data_dir + h5_file_path

        # Open HDF5 file
        with h5.File(h5_file_path, 'r') as file:
            v = np.array(file["X"][:])
            v = torch.Tensor(v)

        # Input
        x_t, fs_s = sf.read(processed_wav_dir + proc_noisy_file_path) # mixture

        T_orig = len(x_t)

        # TF representation (Librosa)
        # Input should be (frames, freq_bins)
        x_tf = stft(x_t,
                fs=fs,
                wlen_sec=wlen_sec,
                win=win,
                hop_percent=hop_percent,
                center=center,
                pad_mode=pad_mode,
                pad_at_end=pad_at_end,
                dtype=dtype) # shape = (freq_bins, frames)

        # Read clean audio
        clean_audio_path = clean_file_path.replace('_' + labels, '')
        clean_audio_path = clean_audio_path.replace('_upsampled', '')
        clean_audio_path = os.path.splitext(clean_audio_path)[0] + '.wav'
        
        # Input
        s_t, fs_s = sf.read(processed_wav_dir + clean_audio_path) 

        # TF representation (Librosa)
        # Input should be (frames, freq_bins)
        s_tf = stft(s_t,
                fs=fs,
                wlen_sec=wlen_sec,
                win=win,
                hop_percent=hop_percent,
                center=center,
                pad_mode=pad_mode,
                pad_at_end=pad_at_end,
                dtype=dtype) # shape = (freq_bins, frames)

        # Reduce frames of audio & label
        if v.shape[-1] < s_tf.shape[-1]:
            x_tf = x_tf[...,:v.shape[-1]]
            s_tf = s_tf[...,:v.shape[-1]]

        X_abs_2 = torch.tensor(np.abs(x_tf)**2, device=device)
        S_abs_2 = torch.tensor(np.abs(s_tf)**2, device=device)

        # Encode-decode
        reconstruction, _, _ = model(torch.t(S_abs_2))
        reconstruction = reconstruction.cpu().numpy()

        # plots of target / estimation
        # Transpose to match librosa.display
        reconstruction = reconstruction.T
        reconstruction = np.sqrt(reconstruction)

        if classif_type == 'oracle':
            # Read labels
            h5_file_path = processed_data_dir + clean_file_path

            with h5.File(h5_file_path, 'r') as file:
                y = np.array(file["Y"][:])

            #y_hat_soft = np.ones(s_tf.shape[1], dtype='float32')[None]
            # y_hat_soft = np.zeros(s_tf.shape[1], dtype='float32')[None]
            y = torch.Tensor(y).to(device)

        # Reduce frames of label
        if v.shape[-1] < y.shape[-1]:
            y = y[...,:v.shape[-1]]

        ## mixture signal (wav + spectro)
        ## target signal (wav + spectro + mask)
        ## estimated signal (wav + spectro + mask)
        signal_list = [
            [s_t, s_tf, None], # clean speech
            # [None, reconstruction, y_hat_hard.cpu().numpy()],
            [None, reconstruction, y.cpu().numpy()],
            [None, reconstruction, y.cpu().numpy()]
        ]
        fig = display_multiple_signals(signal_list,
                            fs=fs, vmin=vmin, vmax=vmax,
                            wlen_sec=wlen_sec, hop_percent=hop_percent,
                            xticks_sec=xticks_sec, fontsize=fontsize)
        
        # put all metrics in the title of the figure
        title = "Input SNR = {:.1f} dB \n" \
            "".format(snr_db)

        fig.suptitle(title, fontsize=40)


        # Save figure
        output_path = output_data_dir + proc_noisy_file_path
        output_path = os.path.splitext(output_path)[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fig.savefig(output_path + '_s_recon.png')
        
        # Clear figure
        plt.close()


        # Encode-decode (Noisy)
        reconstruction, _, _ = model(torch.t(X_abs_2))
        reconstruction = reconstruction.cpu().numpy()

        # plots of target / estimation
        # Transpose to match librosa.display
        reconstruction = reconstruction.T
        reconstruction = np.sqrt(reconstruction)

        ## mixture signal (wav + spectro)
        ## target signal (wav + spectro + mask)
        ## estimated signal (wav + spectro + mask)
        signal_list = [
            [x_t, x_tf, None], # clean speech
            # [None, reconstruction, y_hat_hard.cpu().numpy()],
            [s_t, s_tf, y.cpu().numpy()],
            [None, reconstruction, y.cpu().numpy()]
        ]
        fig = display_multiple_signals(signal_list,
                            fs=fs, vmin=vmin, vmax=vmax,
                            wlen_sec=wlen_sec, hop_percent=hop_percent,
                            xticks_sec=xticks_sec, fontsize=fontsize)
        
        # put all metrics in the title of the figure
        title = "Input SNR = {:.1f} dB \n" \
            "".format(snr_db)

        fig.suptitle(title, fontsize=40)


        # Save figure
        output_path = output_data_dir + proc_noisy_file_path
        output_path = os.path.splitext(output_path)[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fig.savefig(output_path + '_x_recon.png')
        
        # Clear figure
        plt.close()

if __name__ == '__main__':
    main()