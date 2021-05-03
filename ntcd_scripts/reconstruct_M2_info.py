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
from packages.models.models import DeepGenerativeModel_v5

from packages.visualization import display_multiple_signals

##################################### SETTINGS #####################################################

# Dataset
speech_dataset_name = 'ntcd_timit'
noise_dataset_name = 'qutnoise_databases'

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
# M2
if labels == 'vad_labels':
    # model_name = 'ntcd_M2_info_VAD_alpha_10.0_beta_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_192_vloss_388.79'
    # model_name = 'ntcd_M2_info_VAD_alpha_10.0_beta_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_135_vloss_397.11'
    # model_name = 'ntcd_M2_info_VAD_alpha_10.0_beta_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_070_vloss_416.06'
    # model_name = 'ntcd_M2_info_VAD_alpha_0.0_beta_1.0_y_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_101_vloss_401.81'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v2_alpha_10.0_beta_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_189_vloss_407.67'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v2_alpha_0.0_beta_10.0_y_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_140_vloss_419.67'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v2_alpha_0.0_beta_10.0_y_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_208_vloss_411.72'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_10.0_beta_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_144_vloss_396.43'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_0.0_beta_10.0_y_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_172_vloss_401.92'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_0.0_beta_10.0_pretrain_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_145_vloss_398.88'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_10.0_beta_0.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_189_vloss_404.13'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_10.0_beta_10.0_y_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_171_vloss_402.05'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_1.0_beta_1.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_165_vloss_396.70'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v1_alpha_0.0_beta_1.0_y_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_161_vloss_409.56'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_10.0_beta_10.0_gamma_1.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_158_vloss_393.75'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_1.0_beta_10.0_gamma_1.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_179_vloss_388.52'
    model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_10.0_beta_1.0_gamma_1.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_177_vloss_397.30'
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
xticks_sec = 2.0 # in seconds
fontsize = 30

# Data directories
input_speech_dir = os.path.join('data',dataset_size,'raw/')
processed_data_dir = os.path.join('data',dataset_size,'processed/')
processed_wav_dir = os.path.join('data', dataset_size, 'processed', speech_dataset_name, noise_dataset_name + '/')
model_path = os.path.join('models', model_name + '.pt')
output_data_dir = os.path.join('data', dataset_size, 'models', model_name + '/')

def main():
    file = open('output.log','w') 

    print('Torch version: {}'.format(torch.__version__))
    device = torch.device("cuda" if cuda else "cpu")

    print('Device: %s' % (device))
    if torch.cuda.device_count() >= 1: print("Number GPUs: ", torch.cuda.device_count())

    print('Load models')
    model = DeepGenerativeModel_v5([x_dim, y_dim, z_dim, h_dim])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    # model = model.enc_dec_clf # Exclude auxiliary from the model
    
    if cuda: model = model.cuda()

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Create file list
    input_clean_file_paths, \
        output_clean_file_paths = speech_list(input_speech_dir=input_speech_dir,
                            dataset_type=dataset_type)

    print('- Number of test samples: {}'.format(len(output_clean_file_paths)))

    for i, clean_file_path in tqdm(enumerate(output_clean_file_paths)):
        
        # Read video
        h5_file_path = clean_file_path.replace('Clean', 'matlab_raw')
        # h5_file_path = h5_file_path.replace('_' + labels, '')
        h5_file_path = os.path.splitext(h5_file_path)[0] + '_upsampled.h5'
        # h5_file_path = os.path.splitext(h5_file_path)[0] + '_normvideo.h5'
        h5_file_path = processed_data_dir + h5_file_path

        # Open HDF5 file
        with h5.File(h5_file_path, 'r') as file:
            v = np.array(file["X"][:])
            v = torch.Tensor(v)

        # Input
        x_t, fs_s = sf.read(processed_wav_dir + os.path.splitext(clean_file_path)[0] + '_x.wav') # mixture

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

        # Input
        s_t, fs_s = sf.read(processed_wav_dir + os.path.splitext(clean_file_path)[0] + '_s.wav') # mixture

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
        y_hat_soft = model.classify_fromX(torch.t(S_abs_2))
        y_hat_soft = torch.t(y_hat_soft)
        y_hat_hard = (y_hat_soft > 0.5)

        # Encode-decode
        # reconstruction, Z, _, _ = model(torch.t(S_abs_2), torch.t(y_hat_hard))
        reconstruction, Z, _, _ = model(torch.t(S_abs_2), torch.t(y_hat_soft))
        reconstruction = reconstruction.cpu().numpy()

        # plots of target / estimation
        # Transpose to match librosa.display
        reconstruction = reconstruction.T
        reconstruction = np.sqrt(reconstruction)

        if classif_type == 'oracle':
            # Read labels
            h5_file_path = processed_data_dir + os.path.splitext(clean_file_path)[0] + '_' + labels + '.h5'

            with h5.File(h5_file_path, 'r') as file:
                y = np.array(file["Y"][:])

            #y_hat_soft = np.ones(s_tf.shape[1], dtype='float32')[None]
            # y_hat_soft = np.zeros(s_tf.shape[1], dtype='float32')[None]
            y = torch.Tensor(y).to(device)
            y_zeros = torch.zeros_like(y).to(device)
            y_ones = torch.ones_like(y).to(device)

        # Reduce frames of label
        if v.shape[-1] < y.shape[-1]:
            y = y[...,:v.shape[-1]]
            y_zeros = y_zeros[...,:v.shape[-1]]
            y_ones = y_ones[...,:v.shape[-1]]

        # Encode-decode
        reconstruction_oracle, _, _, _ = model(torch.t(S_abs_2), torch.t(y))
        reconstruction_oracle = reconstruction_oracle.cpu().numpy()

        # plots of target / estimation
        # Transpose to match librosa.display
        reconstruction_oracle = reconstruction_oracle.T
        reconstruction_oracle = np.sqrt(reconstruction_oracle)

        ## mixture signal (wav + spectro)
        ## target signal (wav + spectro + mask)
        ## estimated signal (wav + spectro + mask)
        signal_list = [
            [s_t, s_tf, None], # clean speech
            # [None, reconstruction, y_hat_hard.cpu().numpy()],
            [None, reconstruction, y_hat_soft.cpu().numpy()],
            [None, reconstruction_oracle, y.cpu().numpy()]
        ]
        fig = display_multiple_signals(signal_list,
                            fs=fs, vmin=vmin, vmax=vmax,
                            wlen_sec=wlen_sec, hop_percent=hop_percent,
                            xticks_sec=xticks_sec, fontsize=fontsize)
        
        # # put all metrics in the title of the figure
        # title = "Input SNR = {:.1f} dB \n" \
        #     "".format(all_snr_db[i])

        # fig.suptitle(title, fontsize=40)


        # Save figure
        output_path = output_data_dir + clean_file_path
        output_path = os.path.splitext(output_path)[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fig.savefig(output_path + '_s_recon.png')
        
        # Clear figure
        plt.close()


        # Encode-decode (Noisy)
        reconstruction_oracle, _, _, _ = model(torch.t(X_abs_2), torch.t(y))
        reconstruction_oracle = reconstruction_oracle.cpu().numpy()

        # plots of target / estimation
        # Transpose to match librosa.display
        reconstruction_oracle = reconstruction_oracle.T
        reconstruction_oracle = np.sqrt(reconstruction_oracle)

        ## mixture signal (wav + spectro)
        ## target signal (wav + spectro + mask)
        ## estimated signal (wav + spectro + mask)
        signal_list = [
            [x_t, x_tf, None], # clean speech
            # [None, reconstruction, y_hat_hard.cpu().numpy()],
            [s_t, s_tf, y.cpu().numpy()],
            [None, reconstruction_oracle, y.cpu().numpy()]
        ]
        fig = display_multiple_signals(signal_list,
                            fs=fs, vmin=vmin, vmax=vmax,
                            wlen_sec=wlen_sec, hop_percent=hop_percent,
                            xticks_sec=xticks_sec, fontsize=fontsize)
        
        # # put all metrics in the title of the figure
        # title = "Input SNR = {:.1f} dB \n" \
        #     "".format(all_snr_db[i])

        # fig.suptitle(title, fontsize=40)


        # Save figure
        output_path = output_data_dir + clean_file_path
        output_path = os.path.splitext(output_path)[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fig.savefig(output_path + '_x_recon.png')
        
        # Clear figure
        plt.close()




        # Encode-decode (Noisy)
        reconstruction_oracle, _, _, _ = model(torch.t(X_abs_2), torch.t(y_hat_soft))
        reconstruction_oracle = reconstruction_oracle.cpu().numpy()

        # plots of target / estimation
        # Transpose to match librosa.display
        reconstruction_oracle = reconstruction_oracle.T
        reconstruction_oracle = np.sqrt(reconstruction_oracle)

        ## mixture signal (wav + spectro)
        ## target signal (wav + spectro + mask)
        ## estimated signal (wav + spectro + mask)
        signal_list = [
            [x_t, x_tf, None], # clean speech
            # [None, reconstruction, y_hat_hard.cpu().numpy()],
            [s_t, s_tf, y_hat_soft.cpu().numpy()],
            [None, reconstruction_oracle, y_hat_hard.cpu().numpy()]
        ]
        fig = display_multiple_signals(signal_list,
                            fs=fs, vmin=vmin, vmax=vmax,
                            wlen_sec=wlen_sec, hop_percent=hop_percent,
                            xticks_sec=xticks_sec, fontsize=fontsize)
        
        # # put all metrics in the title of the figure
        # title = "Input SNR = {:.1f} dB \n" \
        #     "".format(all_snr_db[i])

        # fig.suptitle(title, fontsize=40)


        # Save figure
        output_path = output_data_dir + clean_file_path
        output_path = os.path.splitext(output_path)[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fig.savefig(output_path + '_x_recon_soft.png')
        
        # Clear figure
        plt.close()




        # Encode-decode (Noisy Ones/Zeros)
        reconstruction_ones, _, _, _ = model(torch.t(X_abs_2), torch.t(y_ones))
        reconstruction_ones = reconstruction_ones.cpu().numpy()

        # plots of target / estimation
        # Transpose to match librosa.display
        reconstruction_ones = reconstruction_ones.T
        reconstruction_ones = np.sqrt(reconstruction_ones)

        ## mixture signal (wav + spectro)
        ## target signal (wav + spectro + mask)
        ## estimated signal (wav + spectro + mask)
        signal_list = [
            [x_t, x_tf, None], # clean speech
            # [None, reconstruction, y_hat_hard.cpu().numpy()],
            [s_t, s_tf, y.cpu().numpy()],
            [None, reconstruction_ones, y_ones.cpu().numpy()]
        ]
        fig = display_multiple_signals(signal_list,
                            fs=fs, vmin=vmin, vmax=vmax,
                            wlen_sec=wlen_sec, hop_percent=hop_percent,
                            xticks_sec=xticks_sec, fontsize=fontsize)
        
        # # put all metrics in the title of the figure
        # title = "Input SNR = {:.1f} dB \n" \
        #     "".format(all_snr_db[i])

        # fig.suptitle(title, fontsize=40)


        # Save figure
        output_path = output_data_dir + clean_file_path
        output_path = os.path.splitext(output_path)[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fig.savefig(output_path + '_x_recon_ones.png')
        
        # Clear figure
        plt.close()


        # Encode-decode (Noisy Ones/Zeros)
        reconstruction_zeros, _, _, _ = model(torch.t(X_abs_2), torch.t(y_zeros))
        reconstruction_zeros = reconstruction_zeros.cpu().numpy()

        # plots of target / estimation
        # Transpose to match librosa.display
        reconstruction_zeros = reconstruction_zeros.T
        reconstruction_zeros = np.sqrt(reconstruction_zeros)

        ## mixture signal (wav + spectro)
        ## target signal (wav + spectro + mask)
        ## estimated signal (wav + spectro + mask)
        signal_list = [
            [x_t, x_tf, None], # clean speech
            # [None, reconstruction, y_hat_hard.cpu().numpy()],
            [s_t, s_tf, y.cpu().numpy()],
            [None, reconstruction_zeros, y_zeros.cpu().numpy()]
        ]
        fig = display_multiple_signals(signal_list,
                            fs=fs, vmin=vmin, vmax=vmax,
                            wlen_sec=wlen_sec, hop_percent=hop_percent,
                            xticks_sec=xticks_sec, fontsize=fontsize)
        
        # # put all metrics in the title of the figure
        # title = "Input SNR = {:.1f} dB \n" \
        #     "".format(all_snr_db[i])

        # fig.suptitle(title, fontsize=40)


        # Save figure
        output_path = output_data_dir + clean_file_path
        output_path = os.path.splitext(output_path)[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fig.savefig(output_path + '_x_recon_zeros.png')
        
        # Clear figure
        plt.close()




        # Show auxiliary classification
        y_hat_soft = model.classify_fromZ(Z)
        y_hat_soft = torch.t(y_hat_soft)
        y_hat_hard = (y_hat_soft > 0.5)

        # Reduce frames of audio & label
        if v.shape[-1] < y_hat_hard.shape[-1]:
            y_hat_soft = y_hat_soft[...,:v.shape[-1]]
            y_hat_hard = y_hat_hard[...,:v.shape[-1]]

        ## mixture signal (wav + spectro)
        ## target signal (wav + spectro + mask)
        ## estimated signal (wav + spectro + mask)
        signal_list = [
            [s_t, s_tf, None], # clean speech
            [None, reconstruction, y_hat_soft.cpu().numpy()],
            [None, reconstruction_oracle, y_hat_hard.cpu().numpy()]
        ]
        fig = display_multiple_signals(signal_list,
                            fs=fs, vmin=vmin, vmax=vmax,
                            wlen_sec=wlen_sec, hop_percent=hop_percent,
                            xticks_sec=xticks_sec, fontsize=fontsize)
        
        # # put all metrics in the title of the figure
        # title = "Input SNR = {:.1f} dB \n" \
        #     "".format(all_snr_db[i])

        # fig.suptitle(title, fontsize=40)
        fig.savefig(output_path + '_recon_aux.png')

        # Clear figure
        plt.close()

if __name__ == '__main__':
    main()