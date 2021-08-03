import sys
sys.path.append('.')

import os
import numpy as np
import torch
import soundfile as sf
import librosa
import json
import matplotlib.pyplot as plt
import concurrent.futures # for multiprocessing
import time
import tempfile
import h5py as h5
from tqdm import tqdm
import math

from packages.processing.stft import stft, istft
from packages.processing.target import clean_speech_IBM, clean_speech_VAD

from packages.metrics import si_sdr_leroux, compute_stats
from pystoi import stoi
from pesq import pesq
from uhh_sp.evaluation import polqa
# from sklearn.metrics import f1_score
from packages.models.utils import f1_loss

from packages.visualization import display_multiple_signals

# Dataset
speech_dataset_name = 'ntcd_timit'
noise_dataset_name = 'ntcd_timit'

if speech_dataset_name == 'ntcd_timit':
    from packages.dataset.ntcd_timit import proc_noisy_clean_pair_dict, speech_list

# Settings
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

## Plot spectrograms
vmin = -40 # in dB
vmax = 20 # in dB
xticks_sec = 0.5 # in seconds
fontsize = 30

## Stats
confidence = 0.95 # confidence interval
eps = 1e-8

# M1/M2
# model_name = 'ntcd_M1_nonorm_hdim_128_128_zdim_016_end_epoch_500/M1_epoch_118_vloss_416.54'
# model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_10.0_beta_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_144_vloss_396.43'
# model_name = 'ntcd_M2_VAD_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_118_vloss_407.90'
model_name = 'ntcd_M2_info_VAD_Lenc_aux_v1_alpha_0.0_beta_10.0_gamma_10.0_y_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_170_vloss_402.17'
# model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_0.0_beta_10.0_y_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_172_vloss_401.92'
# model_name = 'ntcd_M2_info_VAD_alpha_10.0_beta_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_135_vloss_397.11'

# Classifier
classif_name = 'Video_Classifier_vad_noeps_upsampled_resnet_normvideo3_nopretrain_normimage_batch64_noseqlength_end_epoch_100/Video_Net_epoch_007_vloss_4.51'

# Data directories
input_speech_dir = os.path.join('data',dataset_size,'raw/')
processed_data_dir = os.path.join('data',dataset_size,'processed/')
model_data_dir = os.path.join('data', dataset_size, 'models', model_name + '/')
classif_data_dir = os.path.join('data', 'complete', 'models', classif_name + '/')

def compute_metrics_utt(args):
    # Separate args
    i, proc_noisy_file_path, clean_file_path = args[0], args[1], args[2]
    
    # Extract input SNR and noise type
    snr_db = int(proc_noisy_file_path.split('/')[3])
    noise_type = proc_noisy_file_path.split('/')[2]
    speaker = proc_noisy_file_path.split('/')[5]

    # Stationary / Nonstationary noises
    if noise_type in ['Cafe', 'LR', 'Babble', 'Street']:
        noise_stationarity = 'Nonstationary noise'
    if noise_type in ['Car', 'White']:
        noise_stationarity = 'Stationary noise'

    # Read clean audio
    clean_audio_path = clean_file_path.replace('_' + labels, '')
    clean_audio_path = clean_audio_path.replace('_upsampled', '')
    clean_audio_path = os.path.splitext(clean_audio_path)[0] + '.wav'

    # Read files
    s_t, fs_s = sf.read(processed_data_dir + clean_audio_path) # clean speech
    x_t, fs_x = sf.read(processed_data_dir + proc_noisy_file_path) # mixture
    # s_hat_t, fs_s_hat = sf.read(model_data_dir + os.path.splitext(proc_noisy_file_path)[0] + '_s_est.wav') # est. speech
    s_hat_t, fs_s_hat = sf.read(model_data_dir + os.path.splitext(proc_noisy_file_path)[0] + '_s_est_y_hat_hard.wav') # est. speech

    # # Reduce time length if estimated signal is smaller
    # if len(s_hat_t) < len(s_t):
    #     s_t = s_t[:len(s_hat_t)]
    #     x_t = x_t[:len(s_hat_t)]

    # Remove bursts at beginning and end
    offset_time = 0.05
    s_hat_t = s_hat_t[int(offset_time*fs_s):-int(offset_time*fs_s)]
    s_t = s_t[int(offset_time*fs_s):-int(offset_time*fs_s)]
    x_t = x_t[int(offset_time*fs_s):-int(offset_time*fs_s)]

    # # Compute noise as n = x - s
    # n_t = x_t - s_t # noise

    # compute metrics
    ## SI-SDR, SI-SAR, SI-SNR
    # si_sdr, si_sir, si_sar = energy_ratios(s_hat=s_hat_t, s=s_t, n=n_t)
    si_sdr = si_sdr_leroux(s_hat=s_hat_t, s=s_t)
    # si_sdr = si_sdr_leroux(s_hat=x_t, s=s_t)

    ## STOI (or ESTOI?)
    stoi_s_hat = stoi(s_t, s_hat_t, fs, extended=True)
    # stoi_s_hat = stoi(s_t, x_t, fs, extended=True)

    ## PESQ
    pesq_s_hat = pesq(fs, s_t, s_hat_t, 'wb') # wb = wideband
    # pesq_s_hat = pesq(fs, s_t, x_t, 'wb') # wb = wideband
    
    ## POLQA
    # polqa_s_hat = polqa(s, s_t, fs)
    # all_polqa.append(polqa_s_hat)

    ## F1 score
    # ideal binary mask
    h5_file_path = clean_file_path.replace('Clean', 'matlab_raw')
    h5_file_path = h5_file_path.replace('_' + labels, '')
    y_hat_hard = torch.load(classif_data_dir + os.path.splitext(h5_file_path)[0] + '_y_hat_hard.pt', map_location=lambda storage, location: storage) # shape = (frames, freq_bins)
    # y_hat_hard = torch.load(model_data_dir + os.path.splitext(file_path)[0] + '_ibm_soft_est.pt', map_location=lambda storage, location: storage) # shape = (frames, freq_bins)
    # y_hat_hard = y_hat_hard.T # Transpose to match target y, shape = (freq_bins, frames)

    # if labels == 'ibm_labels':
    #     y = clean_speech_IBM(s_tf,
    #                             quantile_fraction=quantile_fraction,
    #                             quantile_weight=quantile_weight)
    if labels == 'vad_labels':
        # Read labels
        h5_file_path = processed_data_dir + clean_file_path

        with h5.File(h5_file_path, 'r') as file:
            y = np.array(file["Y"][:])

        y = torch.LongTensor(y)

    # Convert y to Tensor for f1-score
    y_hat_hard = y_hat_hard.int()
    y = torch.LongTensor(y)

    accuracy, precision, recall, f1score_s_hat = f1_loss(y.flatten(), y_hat_hard.flatten(), epsilon=eps)

    # plots of target / estimation
    # TF representation
    x_tf = stft(x_t,
            fs=fs,
            wlen_sec=wlen_sec,
            win=win,
            hop_percent=hop_percent,
            center=center,
            pad_mode=pad_mode,
            pad_at_end=pad_at_end,
            dtype=dtype) # shape = (freq_bins, frames)

    s_tf = stft(s_t,
            fs=fs,
            wlen_sec=wlen_sec,
            win=win,
            hop_percent=hop_percent,
            center=center,
            pad_mode=pad_mode,
            pad_at_end=pad_at_end,
            dtype=dtype) # shape = (freq_bins, frames)

    s_hat_tf = stft(s_hat_t,
            fs=fs,
            wlen_sec=wlen_sec,
            win=win,
            hop_percent=hop_percent,
            center=center,
            pad_mode=pad_mode,
            pad_at_end=pad_at_end,
            dtype=dtype) # shape = (freq_bins, frames)

    ## mixture signal (wav + spectro)
    ## target signal (wav + spectro + mask)
    ## estimated signal (wav + spectro + mask)
    signal_list = [
        [x_t, x_tf, None], # mixture: (waveform, tf_signal, no mask)
        [s_t, s_tf, y.numpy()], # clean speech
        [s_hat_t, s_hat_tf, y_hat_hard.numpy()]
    ]
    fig = display_multiple_signals(signal_list,
                        fs=fs, vmin=vmin, vmax=vmax,
                        wlen_sec=wlen_sec, hop_percent=hop_percent,
                        xticks_sec=xticks_sec, fontsize=fontsize)
    
    # # put all metrics in the title of the figure
    # title = "Input SNR = {:.1f} dB \n" \
    #     "SI-SDR = {:.1f} dB,  " \
    #     "SI-SIR = {:.1f} dB,  " \
    #     "SI-SAR = {:.1f} dB\n" \
    #     "STOI = {:.2f},  " \
    #     "PESQ = {:.2f} \n" \
    #     "Accuracy = {:.3f},  "\
    #     "Precision = {:.3f},  "\
    #     "Recall = {:.3f},  "\
    #     "F1-score = {:.3f}\n".format(snr_db, si_sdr, si_sir, si_sar, stoi_s_hat, pesq_s_hat,\
    #         accuracy, precision, recall, f1score_s_hat)

    # put all metrics in the title of the figure
    title = "Input SNR = {:.1f} dB \n" \
        "SI-SDR = {:.1f} dB.".format(snr_db, si_sdr)

    fig.suptitle(title, fontsize=40)

    # Save figure
    fig.savefig(model_data_dir + os.path.splitext(proc_noisy_file_path)[0] + '_fig.png')

    # Clear figure
    plt.close()

    metrics = [si_sdr, stoi_s_hat, pesq_s_hat]


    # Vs before / after MCEM
    Vs_bef = torch.load(model_data_dir + os.path.splitext(proc_noisy_file_path)[0] + '_Vs_bef.pt', map_location=lambda storage, location: storage) # shape = (frames, freq_bins)
    Vs_aft = torch.load(model_data_dir + os.path.splitext(proc_noisy_file_path)[0] + '_Vs_aft.pt', map_location=lambda storage, location: storage) # shape = (frames, freq_bins)

    Vs_bef = torch.sqrt(Vs_bef)
    Vs_aft = torch.sqrt(Vs_aft)

    ## mixture signal (wav + spectro)
    ## target signal (wav + spectro + mask)
    ## Vs before MCEM (None + spectro + mask)
    ## Vs after MCEM  (None + spectro + mask)
    signal_list = [
        [x_t, x_tf, None], # mixture: (waveform, tf_signal, no mask)
        [s_t, s_tf, y.numpy()], # clean speech
        [None, Vs_bef, y_hat_hard.numpy()], 
        [None, Vs_aft, y_hat_hard.numpy()]
    ]
    fig = display_multiple_signals(signal_list,
                        fs=fs, vmin=vmin, vmax=vmax,
                        wlen_sec=wlen_sec, hop_percent=hop_percent,
                        xticks_sec=xticks_sec, fontsize=fontsize)

    # put all metrics in the title of the figure
    title = "Input SNR = {:.1f} dB \n" \
        "SI-SDR = {:.1f} dB.".format(snr_db, si_sdr)

    fig.suptitle(title, fontsize=40)

    # Save figure
    fig.savefig(model_data_dir + os.path.splitext(proc_noisy_file_path)[0] + '_Vs.png')

    # Clear figure
    plt.close()

    return metrics, snr_db, noise_type, noise_stationarity, speaker

def main():

    # Dict mapping noisy speech to clean speech
    noisy_clean_pair_paths = proc_noisy_clean_pair_dict(input_speech_dir=processed_data_dir,
                                            dataset_type=dataset_type,
                                            dataset_size=dataset_size,
                                            labels=labels,
                                            upsampled=upsampled)

    # Convert dict to tuples
    args = list(noisy_clean_pair_paths.items())
    
    args = [[i, j[0], j[1]] for i,j in enumerate(args)]
    # args = [[i, j[0], j[1]] for i,j in enumerate(args) if j[0].split('/')[-2] == '54M']
    # args = [[i, j[0], j[1]] for i,j in enumerate(args) if j[0].split('/')[-4] in ['0', '5', '10']]
    
    # args = [[j[0], j[1]] for j in args if j[0].split('/')[-2] == '54M']
    # args = [[j[0], j[1]] for j in args if j[0].split('/')[-4] == '5']
    # args = [[i, j[0], j[1]] for i,j in enumerate(args) if j[0].split('/')[-5] in ['LR']]

    t1 = time.perf_counter()

    all_metrics = []
    all_snr_db = []
    all_noise_types = []
    all_noise_stationarities = []
    all_speakers = []
    for arg in tqdm(args):
        metrics, snr_db, noise_type, noise_stationarity, speaker = compute_metrics_utt(arg)
        all_metrics.append(metrics)
        all_snr_db.append(snr_db)
        all_noise_types.append(noise_type)
        all_noise_stationarities.append(noise_stationarity)
        all_speakers.append(speaker)
    
    # with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
    #     all_results = executor.map(compute_metrics_utt, args)
    
    # # Retrieve metrics and conditions
    # # Transform generator to list
    # all_results = list(all_results)
    # all_metrics = [i[0] for i in all_results]
    # all_snr_db = [i[1] for i in all_results]
    # all_noise_types = [i[2] for i in all_results]
    # all_noise_stationarities = [i[3] for i in all_results]
    # all_speakers = [i[4] for i in all_results]
    
    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')

    # metrics_keys = ['SI-SDR', 'SI-SIR', 'SI-SAR', 'STOI', 'PESQ',\
    #     'Accuracy', 'Precision', 'Recall', 'F1-score']

    # metrics_keys = ['SI-SDR', 'STOI', 'PESQ',\
    #     'Accuracy', 'Precision', 'Recall', 'F1-score']

    metrics_keys = ['SI-SDR', 'STOI', 'PESQ']

    # metrics_keys = ['SI-SDR', 'PESQ',\
    #     'Accuracy', 'Precision', 'Recall', 'F1-score']

    # Compute & save stats
    compute_stats(metrics_keys=metrics_keys,
                  all_metrics=all_metrics,
                  model_data_dir=classif_data_dir,
                  confidence=confidence,
                  all_snr_db=all_snr_db,
                #   all_snr_db=None,
                  all_noise_types=all_noise_types,
                #   all_noise_types=None,
                  all_speakers=all_speakers,
                  all_noise_stationarities=all_noise_stationarities)

def main_polqa():

    # Dict mapping noisy speech to clean speech
    noisy_clean_pair_paths = proc_noisy_clean_pair_dict(input_speech_dir=processed_data_dir,
                                            dataset_type=dataset_type,
                                            dataset_size=dataset_size,
                                            labels=labels,
                                            upsampled=upsampled)

    # Convert dict to tuples
    noisy_clean_pair_paths = list(noisy_clean_pair_paths.items())
    noisy_clean_pair_paths = [[j[0], j[1]] for j in noisy_clean_pair_paths if j[0].split('/')[-4] in ['0', '5', '10']]
    
    v_reference_paths = []
    v_processed_paths = []
    for (proc_noisy_file_path, clean_file_path) in noisy_clean_pair_paths:

        # Read clean audio
        clean_audio_path = clean_file_path.replace('_' + labels, '')
        clean_audio_path = clean_audio_path.replace('_upsampled', '')
        clean_audio_path = os.path.splitext(clean_audio_path)[0] + '.wav'
        v_reference_path = processed_data_dir + clean_audio_path
        v_reference_paths.append(v_reference_path)

        # Read estimated audio
        # v_processed_path = model_data_dir + os.path.splitext(proc_noisy_file_path)[0] + '_s_est.wav'
        v_processed_path = model_data_dir + os.path.splitext(proc_noisy_file_path)[0] + '_s_est_y_hat_hard.wav'
        # v_processed_path = processed_data_dir + proc_noisy_file_path
        v_processed_paths.append(v_processed_path)

    #  POLQA on short audio files
    extended_v_reference_paths = []
    extended_v_processed_paths = []
    
    # Characteristics
    all_snr_db = []
    all_noise_types = []
    all_noise_stationarities = []
    all_speakers = []

    indexes_to_remove = []

    # Fuse both list
    for i, ((proc_noisy_file_path, clean_file_path), v_reference_path, v_processed_path) in enumerate(zip(noisy_clean_pair_paths, v_reference_paths, v_processed_paths)):
        
        # Extract input SNR and noise type
        snr_db = int(proc_noisy_file_path.split('/')[3])
        noise_type = proc_noisy_file_path.split('/')[2]
        speaker = proc_noisy_file_path.split('/')[5]

        # Stationary / Nonstationary noises
        if noise_type in ['Cafe', 'LR', 'Babble', 'Street']:
            noise_stationarity = 'Nonstationary noise'
        if noise_type in ['Car', 'White']:
            noise_stationarity = 'Stationary noise'
                
        # Read files
        s_t, fs_s = sf.read(v_reference_path) # clean speech
        s_hat_t, fs_s_hat = sf.read(v_processed_path) # est. speech

        # if smaller, then convert to numpy array and pad, and remove from list
        if len(s_t) < 3 * fs:
            s_t = np.pad(s_t, (0, 3 * fs - len(s_t)))
            s_hat_t = np.pad(s_hat_t, (0, 3 * fs - len(s_hat_t)))
            
            # Remove from path list
            # v_reference_paths.remove(v_reference_path)
            # v_processed_paths.remove(v_processed_path)
            # del v_reference_paths[i]
            # del v_processed_paths[i]
            indexes_to_remove.append(i)

            # Save as new files
            # Read clean audio
            clean_audio_path = clean_file_path.replace('_' + labels, '')
            clean_audio_path = clean_audio_path.replace('_upsampled', '')
            clean_audio_path = os.path.splitext(clean_audio_path)[0]
            extended_v_reference_path = processed_data_dir + clean_audio_path + '_s_3sec.wav'
            
            # Read estimated audio
            extended_v_processed_path = model_data_dir + os.path.splitext(proc_noisy_file_path)[0] + '_s_est_3sec.wav'

            sf.write(extended_v_reference_path, s_t, fs)
            sf.write(extended_v_processed_path, s_hat_t, fs)

            # Append to extended path list
            extended_v_reference_paths.append(extended_v_reference_path)
            extended_v_processed_paths.append(extended_v_processed_path)
        
        else:
            # Append to list
            all_snr_db.append(snr_db)
            all_noise_types.append(noise_type)
            all_noise_stationarities.append(noise_stationarity)
            all_speakers.append(speaker)

    # # Remove 3rd and last indices from extended list
    # del extended_v_reference_paths[3]
    # del extended_v_reference_paths[-1]

    # del extended_v_processed_paths[3]
    # del extended_v_processed_paths[-1]
    
    for index in sorted(indexes_to_remove, reverse=True):
        del v_reference_paths[index]
        del v_processed_paths[index]

    t1 = time.perf_counter()
    
    # path_all_polqa = polqa(v_reference=v_reference_paths[:1], v_processed=v_processed_paths[:1])
    # extended_all_polqa = polqa(v_reference=extended_v_reference_paths[3:4], v_processed=extended_v_processed_paths[3:4])
    path_all_polqa = polqa(v_reference=v_reference_paths,
                           v_processed=v_processed_paths,
                           narrowband=False, # Reduce computation time
                           wideband=True,
                           n_workers=2)
                        #    n_batches=2)
    # extended_all_polqa = polqa(v_reference=extended_v_reference_paths, v_processed=extended_v_processed_paths)

    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')

    # Transform generator to list
    path_all_polqa = list(path_all_polqa)
    # extended_all_polqa = list(extended_all_polqa)

    with open(model_data_dir + 'path_all_polqa.txt', 'w') as f:
        for item in path_all_polqa:
            f.write("%s\n" % item)

    # with open(model_data_dir + 'extended_all_polqa.txt', 'w') as f:
    #     for item in extended_all_polqa:
    #         f.write("%s\n" % item)

    # Merge lists
    # all_polqa = path_all_polqa + extended_all_polqa
    all_polqa = path_all_polqa
    # all_polqa = extended_all_polqa

    # Flatten list
    all_polqa = [[sub_list[1]] for sub_list in all_polqa]

    # Detect nan
    for i, (polqa_value) in enumerate(all_polqa):
        if math.isnan(polqa_value[0]):
            del all_polqa[i]
            del all_snr_db[i]
            del all_noise_types[i]
            del all_noise_stationarities[i]
            del all_speakers[i]

    metrics_keys = ['POLQA']

    # Compute & save stats
    compute_stats(metrics_keys=metrics_keys,
                  all_metrics=all_polqa,
                  model_data_dir=classif_data_dir,
                  confidence=confidence,
                  all_snr_db=all_snr_db,
                  all_noise_types=all_noise_types,
                  all_speakers=all_speakers,
                  all_noise_stationarities=all_noise_stationarities)
                  
if __name__ == '__main__':
    main()
    # main_polqa()