import sys
sys.path.append('.')

import os
import numpy as np
import torch
import time
import soundfile as sf
from tqdm import tqdm
import librosa
import torchaudio
import matplotlib.pyplot as plt
import h5py as h5

from packages.processing.stft import stft, istft, stft_pytorch
from packages.processing.target import clean_speech_VAD
from packages.models.models import DeepGenerativeModel_v5, DeepGenerativeModel_v6, DeepGenerativeModel_v7, DeepGenerativeModel_v8, DeepGenerativeModel_v9

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
# M2
if labels == 'vad_labels':
    # model_name = 'ntcd_M2_info_VAD_alpha_10.0_beta_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_135_vloss_397.11'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_0.0_beta_10.0_y_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_172_vloss_401.92'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v1_alpha_0.0_beta_10.0_gamma_10.0_y_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_187_vloss_398.45'
    # model_name = 'ntcd_M2_info_VAD_avd_v2_Lenc_aux_v3_alpha_0.0_beta_10.0_gamma_10.0_y_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_148_vloss_406.26'
    # model_name = 'ntcd_M2_info_VAD_avd_v2bis_Lenc_aux_v3_alpha_0.0_beta_10_gamma_10.0_y_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_139_vloss_404.93'
    # model_name = 'ntcd_M2_info_VAD_avd_v3bis_Lenc_aux_v3_alpha_0.0_beta_1.0_gamma_1.0_y_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_155_vloss_0.42'
    # model_name = 'ntcd_M2_info_VAD_avd_v3bis_Lenc_aux_v3_alpha_0.0_beta_1.0_gamma_1.0_y_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_001_vloss_0.86'
    # model_name = 'ntcd_M2_info_VAD_avd_v2quater_Lenc_aux_v3_alpha_0.0_beta_10_gamma_10.0_y_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_167_vloss_382.03'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_0.0_beta_10.0_gamma_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_155_vloss_402.99'
    # model_name = 'ntcd_M2_info_VAD_Uloss_Lenc_aux_v3_alpha_0.0_beta_10.0_gamma_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_144_vloss_397.79'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_0.0_beta_10.0_gamma_10.0_y_float_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_154_vloss_405.56'
    # model_name = 'ntcd_M2_info_VAD_v1bis_Lenc_aux_v3_alpha_1.0_beta_10.0_gamma_10.0_y_float_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_166_vloss_408.41'
    # model_name = 'ntcd_M2_info_VAD_Uloss_Lenc_aux_v3_alpha_1.0_beta_10.0_gamma_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_159_vloss_396.72'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_0.0_beta_10.0_gamma_10.0_0.1y_float_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_152_vloss_411.27'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_1.0_beta_10.0_gamma_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_127_vloss_402.70'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_10.0_beta_10.0_gamma_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_172_vloss_396.21'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_0.0_beta_10.0_gamma_10.0_10.0y_float_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_167_vloss_404.29'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_10.0_beta_10.0_gamma_10.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_148_vloss_402.69'
    # model_name = 'ntcd_M2_info_VAD_v1bis_Lenc_aux_v3_alpha_10.0_beta_10.0_gamma_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_174_vloss_406.48'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_sigmoidinloss_alpha_10.0_beta_10.0_gamma_10.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_130_vloss_406.80'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_10.0_beta_10.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_165_vloss_401.37'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_10.0_beta_10.0_gamma_10.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_144_vloss_403.13'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_20.0_beta_10.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_143_vloss_403.85'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_30.0_beta_10.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_149_vloss_405.09'
    # model_name = 'ntcd_M2_info_VAD_Uloss_Lenc_aux_v3_alpha_10.0_beta_10.0_gamma_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_126_vloss_402.17'
    # model_name = 'ntcd_M2_info_VAD_Uloss_Lenc_aux_v3_alpha_10.0_beta_10.0_gamma_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_153_vloss_397.72'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_100.0_beta_10.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_138_vloss_407.39'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_30.0_beta_10.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_032_end_epoch_500/M2_epoch_158_vloss_407.25'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_30.0_beta_10.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_008_end_epoch_500/M2_epoch_145_vloss_398.98'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_30.0_beta_10.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_008_end_epoch_500/M2_epoch_145_vloss_398.98'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_10.0_beta_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_144_vloss_396.43'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_10.0_beta_1.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_141_vloss_401.92'
    # model_name = 'ntcd_M2_info_VAD_avd_v3_Lenc_aux_v3_alpha_0.0_beta_1.0_gamma_1.0_delta_10.0_y_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_091_vloss_603.01'
    # model_name = 'ntcd_M2_info_VAD_v1ter_Lenc_aux_v3_alpha_0.0_beta_10.0_gamma_10.0_y_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_132_vloss_413.12'
    # model_name = 'ntcd_M2_info_VAD_v1ter_Lenc_aux_v3_alpha_0.0_beta_30.0_gamma_10.0_y_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_130_vloss_408.34'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_10.0_beta_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_144_vloss_396.43'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_10.0_beta_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_180_vloss_458.33'
    # model_name = 'ntcd_M2_info_VAD_v1ter_Lenc_aux_v3_alpha_10.0_beta_10.0_gamma_10.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_160_vloss_405.04'
    # model_name = 'ntcd_M2_info_VAD_v1ter_Lenc_aux_v3_alpha_0.0_beta_20.0_gamma_10.0_y_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_149_vloss_399.84'
    # model_name = 'ntcd_M2_info_VAD_v1ter_Lenc_aux_v3_alpha_10.0_beta_30.0_gamma_30.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_138_vloss_406.50'
    # model_name = 'ntcd_M2_info_VAD_v1ter_Lenc_aux_v3_alpha_10.0_beta_10.0_gamma_1.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_161_vloss_398.66'
    # model_name = 'ntcd_M2_info_VAD_v1ter_Lenc_aux_v3_alpha_1.0_beta_10.0_gamma_1.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_169_vloss_398.30'
    # model_name = 'ntcd_M2_info_VAD_noadv_Lenc_aux_v3_alpha_10.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_108_vloss_416.99'
    # model_name = 'ntcd_M2_info_VAD_v1ter_Lenc_aux_v3_alpha_10.0_beta_30.0_gamma_1.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_125_vloss_403.95'
    # model_name = 'ntcd_M2_info_VAD_Uloss_Lenc_aux_v3_alpha_10.0_beta_10.0_gamma_1.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_109_vloss_404.44'
    # model_name = 'ntcd_M2_info_VAD_Uloss_Lenc_aux_v3_alpha_10.0_beta_30.0_gamma_1.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_151_vloss_379.76'
    # model_name = 'ntcd_M2_info_VAD_Uloss_Lenc_aux_v3_alpha_10.0_beta_30.0_gamma_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_135_vloss_384.82'
    # model_name = 'ntcd_M2_info_VAD_avd_v6_Lenc_aux_v3_alpha_10.0_beta_10_gamma_1.0_y_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_124_vloss_406.34'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_10.0_beta_10.0_gamma_1.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_141_vloss_405.30'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_10.0_beta_1.0_gamma_1.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_122_vloss_408.56'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_10.0_beta_30.0_gamma_10.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_104_vloss_416.66'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_10.0_beta_30.0_gamma_1.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_134_vloss_402.42'
    # model_name = 'ntcd_M2_info_VAD_Uloss_v2_Lenc_aux_v3_alpha_10.0_beta_10.0_gamma_1.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_151_vloss_395.33'
    # model_name = 'ntcd_M2_info_VAD_avd_v2ter_Lenc_aux_v3_alpha_10.0_beta_10.0_gamma_1.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_123_vloss_402.98'
    # model_name = 'ntcd_M2_info_VAD_Uloss_v2_Lenc_aux_v3_alpha_10.0_beta_30.0_gamma_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_172_vloss_383.78'
    # model_name = 'ntcd_M2_info_VAD_Uloss_Lenc_aux_v3_alpha_10.0_beta_40.0_gamma_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_149_vloss_377.87'
    # model_name = 'ntcd_M2_info_VAD_Uloss_v2_Lenc_aux_v3_alpha_10.0_beta_30.0_gamma_20.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_147_vloss_386.49'
    # model_name = 'ntcd_M2_info_VAD_Uloss_v2_Lenc_aux_v3_alpha_10.0_beta_20.0_gamma_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_151_vloss_390.15'
    # model_name = 'ntcd_M2_info_VAD_Uloss_v2_Lenc_aux_v3_alpha_20.0_beta_20.0_gamma_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_149_vloss_390.54'
    # model_name = 'ntcd_M2_info_VAD_avd_v3_Lenc_aux_v3_alpha_0.0_beta_1.0_gamma_1.0_delta_1.0_y_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_159_vloss_406.72'
    # model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_10.0_beta_10.0_gamma_10.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_156_vloss_403.03'
    # model_name = 'ntcd_M2_info_VAD_Uloss_v2_Lenc_aux_v3_alpha_10.0_beta_40.0_gamma_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_133_vloss_379.63'
    # model_name = 'ntcd_M2_info_VAD_Uloss_v2_Lenc_aux_v3_alpha_10.0_beta_20.0_gamma_1.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_139_vloss_390.39'
    # model_name = 'ntcd_M2_info_VAD_v1quater_Lenc_aux_v3_alpha_10.0_beta_10.0_gamma_1.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_172_vloss_408.64'
    # model_name = 'ntcd_M2_info_VAD_Uloss_v3_Lenc_aux_v3_alpha_10.0_beta_10.0_gamma_1.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_146_vloss_397.93'
    # model_name = 'ntcd_M2_info_VAD_Uloss_v3_Lenc_aux_v3_alpha_10.0_beta_10.0_gamma_1.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_146_vloss_402.54'
    model_name = 'ntcd_M2_info_VAD_v1quater_Lenc_aux_v3_alpha_10.0_beta_10.0_gamma_10.0_yhatsoft_nofloat_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_143_vloss_402.58'
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

if classif_type == 'dnn':
    classif_name = 'Video_Classifier_vad_noeps_upsampled_resnet_normvideo3_nopretrain_normimage_batch64_noseqlength_end_epoch_100/Video_Net_epoch_007_vloss_4.51'
    # classif_name = 'AV_Classifier_vad_frozenResNet_upsampled_resnet_normvideo3_nopretrain_normimage_batch64_noseqlength_end_epoch_100/Video_Net_epoch_003_vloss_3.72'

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
classif_data_dir = os.path.join('data', 'complete', 'models', classif_name + '/')

def main():
    file = open('output.log','w') 

    print('Torch version: {}'.format(torch.__version__))
    device = torch.device("cuda" if cuda else "cpu")

    print('Device: %s' % (device))
    if torch.cuda.device_count() >= 1: print("Number GPUs: ", torch.cuda.device_count())

    print('Load models')
    model = DeepGenerativeModel_v5([x_dim, y_dim, z_dim, h_dim])
    # model = DeepGenerativeModel_v6([x_dim, y_dim, z_dim, h_dim])
    # model = DeepGenerativeModel_v7([x_dim, y_dim, z_dim, h_dim])
    # model = DeepGenerativeModel_v8([x_dim, y_dim, z_dim, h_dim])
    # model = DeepGenerativeModel_v9([x_dim, y_dim, z_dim, h_dim])
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
            h5_file_path = processed_data_dir + clean_file_path

            with h5.File(h5_file_path, 'r') as file:
                y = np.array(file["Y"][:])

            #y_hat_soft = np.ones(s_tf.shape[1], dtype='float32')[None]
            # y_hat_soft = np.zeros(s_tf.shape[1], dtype='float32')[None]
            y = torch.Tensor(y).to(device)

        if classif_type == 'dnn':
            # Read labels
            h5_file_path = clean_file_path.replace('Clean', 'matlab_raw')
            h5_file_path = clean_file_path.replace('Clean', 'matlab_raw')
            h5_file_path = h5_file_path.replace('_' + labels, '')
            
            # h5_file_path = proc_noisy_file_path

            h5_file_path = classif_data_dir + h5_file_path
            h5_file_path = os.path.splitext(h5_file_path)[0] + '_y_hat_hard.pt'
            # h5_file_path = os.path.splitext(h5_file_path)[0] + '_y_hat_soft.pt'

            y = torch.load(h5_file_path)
            y = y.float()
            y = y.to(device)
 
        # Z_y = torch.cat([torch.t(Z), y], dim=0)
        Z_y = torch.cat([torch.t(Z), y_hat_soft], dim=0)

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
            # [None, reconstruction_oracle, y.cpu().numpy()],
            # [None, reconstruction_oracle, y_hat_hard.cpu().numpy()]
            [None, reconstruction_oracle, Z_y.cpu().numpy()]
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
        reconstruction, Z, _, _ = model(torch.t(X_abs_2), torch.t(y))
        reconstruction = reconstruction.cpu().numpy()

        Z_y = torch.cat([torch.t(Z), y], dim=0)

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
            [None, reconstruction, Z_y.cpu().numpy()]
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




        # # Encode-decode (Noisy)
        # reconstruction_oracle, _, _, _ = model(torch.t(X_abs_2), torch.t(y_hat_soft))
        # reconstruction_oracle = reconstruction_oracle.cpu().numpy()

        # # plots of target / estimation
        # # Transpose to match librosa.display
        # reconstruction_oracle = reconstruction_oracle.T
        # reconstruction_oracle = np.sqrt(reconstruction_oracle)

        # ## mixture signal (wav + spectro)
        # ## target signal (wav + spectro + mask)
        # ## estimated signal (wav + spectro + mask)
        # signal_list = [
        #     [x_t, x_tf, None], # clean speech
        #     # [None, reconstruction, y_hat_hard.cpu().numpy()],
        #     [s_t, s_tf, y_hat_soft.cpu().numpy()],
        #     [None, reconstruction_oracle, y_hat_hard.cpu().numpy()]
        # ]
        # fig = display_multiple_signals(signal_list,
        #                     fs=fs, vmin=vmin, vmax=vmax,
        #                     wlen_sec=wlen_sec, hop_percent=hop_percent,
        #                     xticks_sec=xticks_sec, fontsize=fontsize)
        
        # # put all metrics in the title of the figure
        # title = "Input SNR = {:.1f} dB \n" \
        #     "".format(snr_db)

        # fig.suptitle(title, fontsize=40)


        # # Save figure
        # output_path = output_data_dir + proc_noisy_file_path
        # output_path = os.path.splitext(output_path)[0]

        # os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # fig.savefig(output_path + '_x_recon_soft.png')
        
        # # Clear figure
        # plt.close()




        # Encode-decode (Noisy Ones/Zeros)
        # reconstruction_ones, _, _, _ = model(torch.t(X_abs_2), torch.t(y_ones))
        reconstruction_ones, _, _, _ = model(torch.t(S_abs_2), torch.t(y_ones))
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
        
        # put all metrics in the title of the figure
        title = "Input SNR = {:.1f} dB \n" \
            "".format(snr_db)

        fig.suptitle(title, fontsize=40)


        # Save figure
        output_path = output_data_dir + clean_file_path
        output_path = os.path.splitext(output_path)[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # fig.savefig(output_path + '_x_recon_ones.png')
        fig.savefig(output_path + '_s_recon_ones.png')
        
        # Clear figure
        plt.close()


        # Encode-decode (Noisy Ones/Zeros)
        # reconstruction_zeros, _, _, _ = model(torch.t(X_abs_2), torch.t(y_zeros))
        reconstruction_zeros, _, _, _ = model(torch.t(S_abs_2), torch.t(y_zeros))
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
        
        # put all metrics in the title of the figure
        title = "Input SNR = {:.1f} dB \n" \
            "".format(snr_db)

        fig.suptitle(title, fontsize=40)


        # Save figure
        output_path = output_data_dir + clean_file_path
        output_path = os.path.splitext(output_path)[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # fig.savefig(output_path + '_x_recon_zeros.png')
        fig.savefig(output_path + '_s_recon_zeros.png')
        
        # Clear figure
        plt.close()




        # Show auxiliary classification
        _, Z, _, _ = model(torch.t(S_abs_2), torch.t(y))
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
            # [None, reconstruction_oracle, y_hat_soft.cpu().numpy()],
            [None, reconstruction_oracle, y_hat_hard.cpu().numpy()]
        ]
        fig = display_multiple_signals(signal_list,
                            fs=fs, vmin=vmin, vmax=vmax,
                            wlen_sec=wlen_sec, hop_percent=hop_percent,
                            xticks_sec=xticks_sec, fontsize=fontsize)
        
        # put all metrics in the title of the figure
        title = "Input SNR = {:.1f} dB \n" \
            "".format(snr_db)

        # fig.suptitle(title, fontsize=40)
        fig.savefig(output_path + '_s_recon_aux.png')

        # Clear figure
        # plt.close()

if __name__ == '__main__':
    main()