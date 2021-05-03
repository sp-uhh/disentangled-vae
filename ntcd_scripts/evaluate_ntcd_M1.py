import sys
sys.path.append('.')

import os
import numpy as np
import torch
import time
import soundfile as sf
import torch.multiprocessing as multiprocessing
import h5py as h5

from packages.processing.stft import stft, istft
from packages.utils import count_parameters
from packages.models.mcem import MCEM_M1
from packages.models.models import VariationalAutoencoder

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

# Hyperparameters 
# M1
model_name = 'ntcd_M1_nonorm_hdim_128_128_zdim_016_end_epoch_500/M1_epoch_118_vloss_416.54'
x_dim = 513 
z_dim = 16
h_dim = [128, 128]
std_norm =False
eps = 1e-8

# NMF
nmf_rank = 10

### MCEM settings
niter = 100 # results reported in the paper were obtained with 500 iterations 
nsamples_E_step = 10
burnin_E_step = 30
nsamples_WF = 25
burnin_WF = 75
var_RW = 0.01

# GPU Multiprocessing
cuda = torch.cuda.is_available()
nb_devices = torch.cuda.device_count()
nb_process_per_device = 2

# Data directories
input_speech_dir = os.path.join('data',dataset_size,'raw/')
processed_data_dir = os.path.join('data',dataset_size,'processed/')
processed_wav_dir = os.path.join('data', dataset_size, 'processed/')
model_path = os.path.join('models', model_name + '.pt')
output_data_dir = os.path.join('data', dataset_size, 'models', model_name + '/')

#####################################################################################################

def process_utt(mcem, model, mean, std,
                proc_noisy_file_path, clean_file_path, device):
    
    # Read video
    h5_file_path = clean_file_path.replace('Clean', 'matlab_raw')
    h5_file_path = h5_file_path.replace('_' + labels, '')
    h5_file_path = processed_data_dir + h5_file_path

    # Open HDF5 file
    with h5.File(h5_file_path, 'r') as file:
        v = np.array(file["X"][:])
        v = torch.Tensor(v)

    # Input
    x_t, fs_x = sf.read(processed_wav_dir + proc_noisy_file_path) # mixture
    
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
    s_t, fs_s = sf.read(processed_wav_dir + clean_audio_path) # mixture
    
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

    # Reduce frames of audio
    if v.shape[-1] < x_tf.shape[-1]:
        x_tf = x_tf[...,:v.shape[-1]]
        s_tf = s_tf[...,:v.shape[-1]]

    # Init MCEM
    #TODO: if std_norm, include mean & std
    mcem.init_parameters(X=x_tf,
                         S=s_tf,
                        vae=model,
                        nmf_rank=nmf_rank,
                        eps=eps,
                        device=device)

    cost = mcem.run()

    S_hat = mcem.S_hat #+ np.finfo(np.float32).eps
    N_hat = mcem.N_hat #+ np.finfo(np.float32).eps

    # ISTFT
    s_hat = istft(S_hat,
                 fs=fs,
                 wlen_sec=wlen_sec,
                 win=win,
                 hop_percent=hop_percent,
                 center=center,
                 max_len=T_orig)

    n_hat = istft(N_hat,
                  fs=fs,
                  wlen_sec=wlen_sec,
                  win=win,
                  hop_percent=hop_percent,
                  center=center,
                  max_len=T_orig)

    # Save .wav files
    output_path = output_data_dir + proc_noisy_file_path
    output_path = os.path.splitext(output_path)[0]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    sf.write(output_path + '_s_est.wav', s_hat, fs)
    sf.write(output_path + '_n_est.wav', n_hat, fs)

    # sf.write(output_path + '_clean_z_s_est.wav', s_hat, fs)
    # sf.write(output_path + '_clean_z_n_est.wav', n_hat, fs)

    # sf.write(output_path + '_clean_z_nomcem_s_est.wav', s_hat, fs)
    # sf.write(output_path + '_clean_z_nomcem_n_est.wav', n_hat, fs)

def process_sublist(device, sublist, mcem, model):

    if cuda: model = model.to(device)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    if std_norm:
        # Load mean and variance        
        audio_h5_dir = os.path.join(processed_data_dir, speech_dataset_name, 'Clean' + '_' + labels + '_upsampled.h5')
        
        # Audio
        with h5.File(audio_h5_dir, 'r') as file:
            mean = file['X_train_mean'][:]
            std = file['X_train_std'][:]

        mean = torch.tensor(mean).to(device)
        std = torch.tensor(std).to(device)
    else:
        mean = None
        std = None
    
    for (proc_noisy_file_path, clean_file_path) in sublist:
        process_utt(mcem, model, mean, std, proc_noisy_file_path, clean_file_path, device)

def main():
    file = open('output.log','w') 

    print('Torch version: {}'.format(torch.__version__))

    # Start context for GPU multiprocessing
    ctx = multiprocessing.get_context('spawn')

    print('Load models')
    model = VariationalAutoencoder([x_dim, z_dim, h_dim])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    print('- Number of learnable parameters: {}'.format(count_parameters(model)))
    
    mcem = MCEM_M1(niter=niter,
                   nsamples_E_step=nsamples_E_step,
                   burnin_E_step=burnin_E_step, nsamples_WF=nsamples_WF, 
                   burnin_WF=burnin_WF, var_RW=var_RW)

    # Dict mapping noisy speech to clean speech
    noisy_clean_pair_paths = proc_noisy_clean_pair_dict(input_speech_dir=processed_data_dir,
                                            dataset_type=dataset_type,
                                            dataset_size=dataset_size,
                                            labels=labels,
                                            upsampled=upsampled)

    # Convert dict to tuples
    noisy_clean_pair_paths = list(noisy_clean_pair_paths.items())

    # Split list in nb_devices * nb_processes_per_device
    b = np.array_split(noisy_clean_pair_paths, nb_devices*nb_process_per_device)
    
    # Assign each list to a process
    b = [(i%nb_devices, sublist, mcem, model) for i, sublist in enumerate(b)]    # Split list in nb_devices * nb_processes_per_device

    print('Start evaluation')
    # start = time.time()
    t1 = time.perf_counter()
    
    with ctx.Pool(processes=nb_process_per_device*nb_devices) as multi_pool:
        multi_pool.starmap(process_sublist, b)
    
    # # Test script on 1 sublist
    # process_sublist(*b[0])

    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')

if __name__ == '__main__':
    main()