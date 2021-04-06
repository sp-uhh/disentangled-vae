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
noise_dataset_name = 'qutnoise_databases'

if speech_dataset_name == 'ntcd_timit':
    from packages.dataset.ntcd_timit import proc_noisy_clean_pair_dict, speech_list

dataset_type = 'test'

dataset_size = 'subset'
# dataset_size = 'complete'
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
model_name = 'ntcd_M1_nonorm_hdim_128_128_zdim_016_end_epoch_500/M1_epoch_164_vloss_408.82'
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
nb_process_per_device = 1

# Data directories
input_speech_dir = os.path.join('data',dataset_size,'raw/')
processed_data_dir = os.path.join('data',dataset_size,'processed/')
processed_wav_dir = os.path.join('data', dataset_size, 'processed', speech_dataset_name, noise_dataset_name + '/')
model_path = os.path.join('models', model_name + '.pt')
output_data_dir = os.path.join('data', dataset_size, 'models', model_name + '/')

#####################################################################################################

def process_utt(mcem, model, mean, std, clean_file_path, device):
    
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
    x_t, fs_x = sf.read(processed_wav_dir + os.path.splitext(clean_file_path)[0] + '_x.wav') # mixture
    
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
    #TODO: convert to tensor

    # ISTFT
    # S_hat = S_hat[None] # 1channel
    # s_hat = istft_pytorch(S_hat,
    s_hat = istft(S_hat,
                 fs=fs,
                 wlen_sec=wlen_sec,
                 win=win,
                 hop_percent=hop_percent,
                 center=center,
                 max_len=T_orig)

    # N_hat = N_hat[None] # 1channel
    n_hat = istft(N_hat,
                  fs=fs,
                  wlen_sec=wlen_sec,
                  win=win,
                  hop_percent=hop_percent,
                  center=center,
                  max_len=T_orig)

    # Save .wav files
    output_path = output_data_dir + clean_file_path
    output_path = os.path.splitext(output_path)[0]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # sf.write(output_path + '_s_est.wav', s_hat, fs)
    # sf.write(output_path + '_n_est.wav', n_hat, fs)

    # sf.write(output_path + '_clean_z_s_est.wav', s_hat, fs)
    # sf.write(output_path + '_clean_z_n_est.wav', n_hat, fs)

    sf.write(output_path + '_clean_z_nomcem_s_est.wav', s_hat, fs)
    sf.write(output_path + '_clean_z_nomcem_n_est.wav', n_hat, fs)

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
    
    for file_path in sublist:
        process_utt(mcem, model, mean, std, file_path, device)

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

    # Create file list
    input_clean_file_paths, \
        output_clean_file_paths = speech_list(input_speech_dir=input_speech_dir,
                            dataset_type=dataset_type)

    print('- Number of test samples: {}'.format(len(output_clean_file_paths)))

    # Split list in nb_devices * nb_processes_per_device
    b = np.array_split(output_clean_file_paths, nb_devices*nb_process_per_device)
    
    # Assign each list to a process
    b = [(i%nb_devices, sublist, mcem, model) for i, sublist in enumerate(b)]

    print('Start evaluation')
    # start = time.time()
    t1 = time.perf_counter()
    
    # with ctx.Pool(processes=nb_process_per_device*nb_devices) as multi_pool:
    #     multi_pool.starmap(process_sublist, b)
    
    # Test script on 1 sublist
    process_sublist(*b[0])

    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')

if __name__ == '__main__':
    main()