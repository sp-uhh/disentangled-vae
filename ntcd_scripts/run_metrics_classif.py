import sys
sys.path.append('.')

import os
import numpy as np
import torch, torchaudio
import librosa
import matplotlib.pyplot as plt
import concurrent.futures # for multiprocessing
import time
import tempfile
import ffmpeg
import skvideo.io
import cv2
import h5py as h5
import math

from packages.metrics import energy_ratios, compute_stats

from packages.processing.stft import stft_pytorch
from packages.visualization import display_multiple_signals
from packages.models.utils import f1_loss

dataset_name = 'ntcd_timit'
if dataset_name == 'ntcd_timit':
    from packages.dataset.ntcd_timit import proc_noisy_clean_pair_dict, speech_list

# Dataset
# dataset_size = 'subset'
dataset_size = 'complete'

dataset_type = 'test'
# dataset_type = 'validation'

# Labels
labels = 'vad_labels'
# labels = 'ibm_labels'
upsampled = True

# ## Video
# visual_frame_rate_i = 30 # initial visual frames per second

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

## Plot spectrograms
vmin = -40 # in dB
vmax = 20 # in dB
xticks_sec = 2.0 # in seconds
fontsize = 30

## Stats
#TODO: compute median
confidence = 0.95 # confidence interval

## Classifier
if labels == 'vad_labels':
    # classif_name = 'Audio_Classifier_vad_loss_eps_upsampled_align_shuffle_nopretrain_normdataset_batch64_noseqlength_end_epoch_100/Video_Net_epoch_009_vloss_16.64'
    # classif_name = 'AV_Classifier_vad_mcb_nopretrain_normdataset_batch64_noseqlength_end_epoch_100/Video_Net_epoch_002_vloss_3.85'
    # classif_name = 'AV_Classifier_vad_mcb_nopretrain_normdataset_batch64_noseqlength_end_epoch_100/Video_Net_epoch_001_vloss_4.96'
    # classif_name = 'AV_Classifier_vad_noeps_upsampled_resnet_normvideo3_nopretrain_normimage_batch64_noseqlength_end_epoch_100/Video_Net_epoch_001_vloss_4.52'
    # classif_name = 'AV_Classifier_vad_mcb_upsampled_resnet_normvideo3_nopretrain_normimage_batch64_noseqlength_end_epoch_100/Video_Net_epoch_001_vloss_3.89'
    # classif_name = 'AV_Classifier_vad_cleanspeech_resnet_normvideo3_nopretrain_normimage_batch64_noseqlength_end_epoch_100/Video_Net_epoch_004_vloss_1.15'
    # classif_name = 'Audio_Classifier_vad_cleanspeech_align_shuffle_nopretrain_normdataset_batch64_noseqlength_end_epoch_100/Video_Net_epoch_006_vloss_1.13'
    # classif_name = 'AV_Classifier_vad_frozenResNet_upsampled_resnet_normvideo3_nopretrain_normimage_batch64_noseqlength_end_epoch_100/Video_Net_epoch_003_vloss_3.72'
    # classif_name = 'Audio_Classifier_vad_loss_eps_upsampled_align_shuffle_nopretrain_normdataset_batch64_noseqlength_end_epoch_100/Video_Net_epoch_008_vloss_4.07'
    classif_name = 'AV_Classifier_vad_frozenResNet_mcb_upsampled_resnet_normvideo3_nopretrain_normimage_batch64_noseqlength_end_epoch_100/Video_Net_epoch_005_vloss_3.85'

if labels == 'ibm_labels':
    classif_name = 'Audio_Classifier_ibm_normdataset_batch16_noseqlength_end_epoch_100/Video_Net_epoch_006_vloss_9.22'

# Data directories
processed_data_dir = os.path.join('data',dataset_size,'processed/')
classif_data_dir = os.path.join('data', dataset_size, 'models', classif_name + '/')

####################################################

def compute_metrics_utt(args):
    # Separate args
    i, proc_noisy_file_path, clean_file_path = args[0], args[1], args[2]

    # Extract input SNR and noise type
    snr_db = int(proc_noisy_file_path.split('/')[3])
    noise_type = proc_noisy_file_path.split('/')[2]
    speaker = proc_noisy_file_path.split('/')[5]

    # snr_db = 100
    # noise_type = 'None'
    # speaker = proc_noisy_file_path.split('/')[3]

    # Read target
    h5_file_path = processed_data_dir + clean_file_path
    # h5_file_path = processed_data_dir + os.path.splitext(clean_file_path)[0] + '_' + labels + '.h5'

    with h5.File(h5_file_path, 'r') as file:
        y = np.array(file["Y"][:])
        y = torch.LongTensor(y) # Convert y to Tensor for f1-score
    
    # Output file
    output_path = classif_data_dir + proc_noisy_file_path
    # output_path = classif_data_dir + clean_file_path
    output_path = os.path.splitext(output_path)[0]

    # Load y_hat_hard / y_hat_soft
    y_hat_hard = torch.load(output_path + '_y_hat_hard.pt')
    y_hat_soft = torch.load(output_path + '_y_hat_soft.pt')

    if y.shape[-1] != y_hat_hard.shape[-1]:
        print(i)

    ## F1 score
    accuracy, precision, recall, f1score_s_hat = f1_loss(y.flatten(), y_hat_hard.flatten(), epsilon=eps)

    # Convert to float
    accuracy = accuracy.item()
    precision = precision.item()
    recall = recall.item()
    f1score_s_hat = f1score_s_hat.item()

    # Clean wav path
    clean_wav_path = os.path.splitext(clean_file_path)[0]
    clean_wav_path = clean_wav_path.replace('_' + labels, '')
    if upsampled:
        clean_wav_path = clean_wav_path.replace('_upsampled', '')
    clean_wav_path = clean_wav_path + '.wav'

    # Read files
    s_t, fs_s = torchaudio.load(processed_data_dir + clean_wav_path)
    s_t = s_t[0] # 1channel
    x_t, fs_x = torchaudio.load(processed_data_dir + proc_noisy_file_path)
    # x_t, fs_x = torchaudio.load(processed_data_dir + clean_file_path)
    x_t = x_t[0] # 1channel

    # x = x/np.max(x)
    T_orig = len(x_t)

    # Normalize audio
    norm_max = torch.max(torch.abs(x_t))
    x_t = x_t / norm_max
    
    # TF representation (PyTorch)
    # Input should be (frames, freq_bins)
    x_tf = stft_pytorch(x_t,
            fs=fs,
            wlen_sec=wlen_sec,
            win=win, 
            hop_percent=hop_percent,
            center=center,
            pad_mode=pad_mode,
            pad_at_end=pad_at_end) # shape = (freq_bins, frames)

    # Real + j * Img
    x_tf = x_tf[...,0].numpy() + 1j * x_tf[...,1].numpy()

    # plots of target / estimation
    # Normalize audio
    s_t = s_t / norm_max

    # TF representation (PyTorch)
    s_tf = stft_pytorch(s_t,
            fs=fs,
            wlen_sec=wlen_sec,
            win=win, 
            hop_percent=hop_percent,
            center=center,
            pad_mode=pad_mode,
            pad_at_end=pad_at_end) # shape = (freq_bins, frames)

    # Real + j * Img
    s_tf = s_tf[...,0].numpy() + 1j * s_tf[...,1].numpy()

    # Reduce frames of audio
    if y.shape[-1] < x_tf.shape[-1]:
        x_tf = x_tf[...,:y.shape[-1]]
        s_tf = s_tf[...,:y.shape[-1]]

    ## mixture signal (wav + spectro)
    ## target signal (wav + spectro + mask)
    ## estimated signal (wav + spectro + mask)
    signal_list = [
        [x_t.numpy(), x_tf, None], # mixture: (waveform, tf_signal, no mask)
        [s_t.numpy(), s_tf, y.numpy()], # clean speech
        [None, y_hat_soft.numpy(), y_hat_hard.numpy()]
    ]

    fig = display_multiple_signals(signal_list,
                        fs=fs, vmin=vmin, vmax=vmax,
                        wlen_sec=wlen_sec, hop_percent=hop_percent,
                        xticks_sec=xticks_sec, fontsize=fontsize,
                        last_only_label=True)
    
    # put all metrics in the title of the figure
    title = "Input SNR = {:.1f} dB \n" \
        "Noise type = {}, \n"\
        "Accuracy = {:.3f},  "\
        "Precision = {:.3f},  "\
        "Recall = {:.3f},  "\
        "F1-score = {:.3f}\n".format(snr_db, noise_type,\
            accuracy, precision, recall, f1score_s_hat)

    fig.suptitle(title, fontsize=40)

    # Save figure
    output_path = classif_data_dir + proc_noisy_file_path
    # output_path = classif_data_dir + clean_file_path
    output_path = os.path.splitext(output_path)[0]

    fig.savefig(output_path + '_hard_mask.png')
    
    # Clear figure
    plt.close()


    # # make video with audio with target
    # # Create temporary file for video without audio
    # with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp:
    #     out = skvideo.io.FFmpegWriter(tmp.name,
    #             inputdict={'-r': str(visual_frame_rate_o),
    #                     '-s':'{}x{}'.format(width,height)},
    #             outputdict={'-filter:v': 'fps=fps={}'.format(visual_frame_rate_o),
    #                         '-c:v': 'libx264',
    #                         '-crf': str(crf),
    #                         '-preset': 'veryslow'}
    #     )

    #     # Write video
    #     for j, x_video_frame in enumerate(x_video.T):
    #         # Add label on the video
    #         if y[...,j] == 1:
    #             x_video_frame.T[-9:,-9:] = 255 # white square
    #         out.writeFrame(x_video_frame.T)
            
    #     # close out the video writer
    #     out.close()

    #     # Add the audio using ffmpeg-python
    #     video = ffmpeg.input(tmp.name)
    #     audio = ffmpeg.input(input_speech_dir + audio_file_path)
    #     out = ffmpeg.output(video, audio, os.path.splitext(output_path)[0] + '_oracle_audio.mp4', vcodec='copy', acodec='aac', strict='experimental')
    #     out = out.overwrite_output()
    #     out.run()

    # # make video with audio with pred
    # # Create temporary file for video without audio
    # with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp:
    #     out = skvideo.io.FFmpegWriter(tmp.name,
    #             inputdict={'-r': str(visual_frame_rate_o),
    #                     '-s':'{}x{}'.format(width,height)},
    #             outputdict={'-filter:v': 'fps=fps={}'.format(visual_frame_rate_o),
    #                         '-c:v': 'libx264',
    #                         '-crf': str(crf),
    #                         '-preset': 'veryslow'}
    #     )

    #     # Write video
    #     for j, x_video_frame in enumerate(x_video.T):
    #         # Add label on the video
    #         if y_hat_hard[...,j] == 1:
    #             x_video_frame.T[-9:,-9:] = 255 # white square
    #         out.writeFrame(x_video_frame.T)
        
    #     # close out the video writer
    #     out.close()

    #     # Add the audio using ffmpeg-python
    #     video = ffmpeg.input(tmp.name)
    #     audio = ffmpeg.input(input_speech_dir + audio_file_path)
    #     out = ffmpeg.output(video, audio, os.path.splitext(output_path)[0] + '_pred_audio.mp4', vcodec='copy', acodec='aac', strict='experimental')
    #     out = out.overwrite_output()
    #     out.run()

    #TODO: make video with y_hat_soft

    metrics = [accuracy, precision, recall, f1score_s_hat]

    return metrics, snr_db, noise_type, speaker

def main():

    # Dict mapping noisy speech to clean speech
    noisy_clean_pair_paths = proc_noisy_clean_pair_dict(input_speech_dir=processed_data_dir,
                                            dataset_type=dataset_type,
                                            dataset_size=dataset_size,
                                            labels=labels,
                                            upsampled=upsampled)

    # Convert dict to tuples
    args = list(noisy_clean_pair_paths.items())

    # # # TODO: correct audio / target alignment (paths not matching)
    # input_clean_file_paths, \
    #     output_clean_file_paths = speech_list(input_speech_dir='data/complete/raw/',
    #                         dataset_type=dataset_type)

    # noisy_clean_pair_paths = [(input_clean_file_path, output_clean_file_path)
    #                 for input_clean_file_path, output_clean_file_path\
    #                     in zip(input_clean_file_paths, output_clean_file_paths)]

    # # Convert dict to tuples
    # args = noisy_clean_pair_paths
    
    args = [[i, j[0], j[1]] for i,j in enumerate(args)]

    t1 = time.perf_counter()

    # all_metrics = []
    # all_snr_db = []
    # all_noise_types = []
    # all_speakers = []
    # for arg in args:
    #     metrics, snr_db, noise_type, speaker = compute_metrics_utt(arg)
    #     all_metrics.append(metrics)
    #     all_snr_db.append(snr_db)
    #     all_noise_types.append(noise_type)
    #     all_speakers.append(speaker)

    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        all_results = executor.map(compute_metrics_utt, args)
    
    # Retrieve metrics and conditions
    # Transform generator to list
    all_results = list(all_results)
    all_metrics = [i[0] for i in all_results]
    all_snr_db = [i[1] for i in all_results]
    all_noise_types = [i[2] for i in all_results]
    all_speakers = [i[3] for i in all_results]

    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')


    metrics_keys = ['Accuracy', 'Precision', 'Recall', 'F1-score']

    # Compute & save stats
    compute_stats(metrics_keys=metrics_keys,
                  all_metrics=all_metrics,
                  model_data_dir=classif_data_dir,
                  confidence=confidence,
                  all_snr_db=all_snr_db,
                #   all_snr_db=None,
                  all_noise_types=all_noise_types,
                #   all_noise_types=None,
                  all_speakers=all_speakers)

if __name__ == '__main__':
    main()