import sys
sys.path.append('.')

import os
import torch
import pickle
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import h5py as h5
import math

from torch.utils.data import DataLoader
# from video_net import VideoClassifier
# from data_handling import VideoFrames

from packages.data_handling import NoisyWavWholeSequenceSpectrogramLabeledFrames, NoisyWavWholeSequenceWavLabeledFrames
from packages.models.Audio_Net import DeepVAD_audio
from packages.models.utils import binary_cross_entropy, f1_loss
from packages.utils import count_parameters, collate_many2many_audio, collate_many2many_audio_waveform
from packages.processing.stft import stft_pytorch

# Dataset
# dataset_size = 'subset'
dataset_size = 'complete'
upsampled = True

dataset_name = 'ntcd_timit'

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

# System 
cuda = torch.cuda.is_available()
cuda_device = "cuda:0"
device = torch.device(cuda_device if cuda else "cpu")
num_workers = 16
pin_memory = True
non_blocking = True
rdcc_nbytes = 1024**2*400  # The number of bytes to use for the chunk cache
                           # Default is 1 Mb
                           # Here we are using 400Mb of chunk_cache_mem here
rdcc_nslots = 1e5 # The number of slots in the cache's hash table
                  # Default is 521
                  # ideally 100 x number of chunks that can be fit in rdcc_nbytes
                  # (see https://docs.h5py.org/en/stable/high/file.html?highlight=rdcc#chunk-cache)

# Deep Generative Model
x_dim = 513 
if labels == 'vad_labels':
    y_dim = 1
if labels == 'ibm_labels':
    y_dim = 513
# h_dim = [128, 128]
lstm_layers = 2
lstm_hidden_size = 1024 
batch_norm=False
std_norm =True
eps = 1e-8

# Training
# batch_size = 64
batch_size = 16
# batch_size = 2
learning_rate = 1e-4
# weight_decay = 1e-4
# momentum = 0.9
log_interval = 1
start_epoch = 1
end_epoch = 100

if labels == 'vad_labels':
    # model_name = 'Audio_Classifier_vad_upsampled_align_shuffle_nopretrain_normdataset_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'Audio_Classifier_vad_cleanspeech_align_shuffle_nopretrain_normdataset_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'Audio_Classifier_vad_cleanspeech_upsampled_align_shuffle_nopretrain_normdataset_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    model_name = 'Audio_Classifier_vad_loss_eps_upsampled_align_shuffle_nopretrain_normdataset_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)

if labels == 'ibm_labels':
    model_name = 'Audio_Classifier_ibm_normdataset_batch16_noseqlength_end_epoch_{:03d}'.format(end_epoch)

# Data directories
input_video_dir = os.path.join('data', dataset_size, 'processed/')
# output_h5_dir = input_video_dir + os.path.join(dataset_name, 'Noisy', dataset_name + '_' + 'power_spec' + '_statistics.h5')
# output_h5_dir = input_video_dir + os.path.join(dataset_name, 'Clean', dataset_name + '_' + 'power_spec' + '_statistics.h5')
# output_h5_dir = input_video_dir + os.path.join(dataset_name, 'Clean', dataset_name + '_' + 'log_power_spec' + '_statistics.h5')
# output_h5_dir = input_video_dir + os.path.join(dataset_name, 'Clean', dataset_name + '_' + 'log_power_spec_upsampled' + '_statistics.h5')
output_h5_dir = input_video_dir + os.path.join(dataset_name, 'Noisy', dataset_name + '_' + 'log_power_spec_upsampled' + '_statistics.h5')

#####################################################################################################

print('Load data')
train_dataset = NoisyWavWholeSequenceSpectrogramLabeledFrames(input_video_dir=input_video_dir, dataset_type='train',
                                                              dataset_size=dataset_size, labels=labels, upsampled=upsampled,
                                                              fs=fs, wlen_sec=wlen_sec, win=win, hop_percent=hop_percent,
                                                              center=center, pad_mode=pad_mode, pad_at_end=pad_at_end, eps=eps)
valid_dataset = NoisyWavWholeSequenceSpectrogramLabeledFrames(input_video_dir=input_video_dir, dataset_type='validation',
                                                              dataset_size=dataset_size, labels=labels, upsampled=upsampled,
                                                              fs=fs, wlen_sec=wlen_sec, win=win, hop_percent=hop_percent,
                                                              center=center, pad_mode=pad_mode, pad_at_end=pad_at_end, eps=eps)                                                              

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=None, 
                        batch_sampler=None, num_workers=num_workers, pin_memory=pin_memory, 
                        drop_last=False, timeout=0, worker_init_fn=None, collate_fn=collate_many2many_audio)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, sampler=None, 
                        batch_sampler=None, num_workers=num_workers, pin_memory=pin_memory, 
                        drop_last=False, timeout=0, worker_init_fn=None, collate_fn=collate_many2many_audio)

# train_dataset = NoisyWavWholeSequenceWavLabeledFrames(input_video_dir=input_video_dir, dataset_type='train',
#                                                               dataset_size=dataset_size, labels=labels,
#                                                               fs=fs, wlen_sec=wlen_sec, win=win, hop_percent=hop_percent,
#                                                               center=center, pad_mode=pad_mode, pad_at_end=pad_at_end, eps=eps)
# valid_dataset = NoisyWavWholeSequenceWavLabeledFrames(input_video_dir=input_video_dir, dataset_type='validation',
#                                                               dataset_size=dataset_size, labels=labels,
#                                                               fs=fs, wlen_sec=wlen_sec, win=win, hop_percent=hop_percent,
#                                                               center=center, pad_mode=pad_mode, pad_at_end=pad_at_end, eps=eps)                                                              

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=None, 
#                         batch_sampler=None, num_workers=num_workers, pin_memory=pin_memory, 
#                         drop_last=False, timeout=0, worker_init_fn=None, collate_fn=collate_many2many_audio_waveform)
# valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, sampler=None, 
#                         batch_sampler=None, num_workers=num_workers, pin_memory=pin_memory, 
#                         drop_last=False, timeout=0, worker_init_fn=None, collate_fn=collate_many2many_audio_waveform)

print('- Number of training samples: {}'.format(len(train_dataset)))
print('- Number of validation samples: {}'.format(len(valid_dataset)))

print('- Number of training batches: {}'.format(len(train_loader)))
print('- Number of validation batches: {}'.format(len(valid_loader)))

def main():
    print('Create model')
    model = DeepVAD_audio(lstm_layers, lstm_hidden_size, y_dim)

    if cuda: model = model.to(device)

    model = nn.parallel.DataParallel(model, device_ids=[0,1,2,3])
    # model = nn.parallel.DataParallel(model, device_ids=[4,5,6,7])

    if cuda: model = model.to(device)

    nfft = int(wlen_sec * fs) # STFT window length in samples
    window = torch.hann_window(window_length=nfft).to(device)

    # Create model folder
    model_dir = os.path.join('models', model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if std_norm:
        print('Load mean and std')
        with h5.File(output_h5_dir, 'r') as file:
            mean = file['X_train_mean'][:]
            std = file['X_train_std'][:]

        # Save mean and std
        #TODO: copy h5py file
        np.save(model_dir + '/' + 'trainset_mean.npy', mean)
        np.save(model_dir + '/' + 'trainset_std.npy', std)

        mean = torch.tensor(mean).to(device)
        std = torch.tensor(std).to(device)

    # Start log file
    file = open(model_dir + '/' +'output_batch.log','w') 
    file = open(model_dir + '/' +'output_epoch.log','w') 

    # Optimizer settings
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    # criterion = nn.CrossEntropyLoss().to(device) # Ignore padding in the loss

    t = len(train_loader)
    m = len(valid_loader)
    print('- Number of learnable parameters: {}'.format(count_parameters(model)))

    print('Start training')
    for epoch in range(start_epoch, end_epoch):
        model.train()
        total_loss, total_accuracy, total_precision, total_recall, total_f1_score = (0, 0, 0, 0, 0)
        # for batch_idx, (x, y) in enumerate(train_loader):
        for batch_idx, (lengths, x, y) in tqdm(enumerate(train_loader)):
            if cuda:
                # x, y, lengths = x.to(device, non_blocking=non_blocking), y.long().to(device, non_blocking=non_blocking), lengths.to(device, non_blocking=non_blocking)
                x, y, lengths = x.to(device, non_blocking=non_blocking),\
                                    y.long().to(device, non_blocking=non_blocking),\
                                        lengths.to(device, non_blocking=non_blocking)

            # # TF representation (PyTorch)
            # x = stft_pytorch(x,
            #         fs=fs,
            #         wlen_sec=wlen_sec,
            #         win=window, 
            #         hop_percent=hop_percent,
            #         center=center,
            #         pad_mode=pad_mode,
            #         pad_at_end=pad_at_end) # shape = (freq_bins, frames)

            # # Power spectrogram
            # x = x[...,0]**2 + x[...,1]**2

            # # Apply log
            # x = torch.log(x + eps)

            # # Swap x_dim and seq_length axes
            # x = x.transpose(1,-1)

            # Normalize power spectrogram
            if std_norm:
                x_norm = x - mean.T
                x_norm /= (std + eps).T

                y_hat_soft = model(x_norm, lengths) 
            else:
                y_hat_soft = model(x, lengths)

            # y_hat_soft = torch.squeeze(y_hat_soft)
            loss = 0.
            for (length, pred, target) in zip(lengths, y_hat_soft, y):
                # loss += binary_cross_entropy(pred[:length], target[:length])
                loss += binary_cross_entropy(pred[:length], target[:length], eps)
            # loss /= len(lengths)
            # loss = binary_cross_entropy(y_hat_soft, y, eps)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            # _, y_hat_hard = torch.max(y_hat_soft.data, 1)
            y_hat_soft = torch.sigmoid(y_hat_soft.detach())
            y_hat_hard = (y_hat_soft > 0.5).int()
            # y_hat_hard = (y_hat_soft.detach() > 0.5).int()

            # exclude padding from F1-score
            y_hat_hard_batch, y_batch = [], []
            accuracy, precision, recall, f1_score = 0., 0., 0., 0.
            for (length, pred, target) in zip(lengths, y_hat_hard, y):
                acc, prec, rec, f1 = f1_loss(y_hat_hard=torch.flatten(pred[:length]),
                                                                y=torch.flatten(target[:length]),
                                                                epsilon=eps)
                accuracy += acc
                precision += prec
                recall += rec
                f1_score += f1
            accuracy /= len(lengths)
            precision /= len(lengths)
            recall /= len(lengths)
            f1_score /= len(lengths)

            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1_score += f1_score

            # Save to log
            if batch_idx % log_interval == 0:
                print(('Train Epoch: {:2d}   [{:4d}/{:4d} ({:2d}%)]    Loss: {:.2f}    Accuracy: {:.2f}    Precision: {:.2f}    Recall: {:.2f}    F1-score.: {:.2f}'\
                    + '').format(epoch, batch_idx*len(x), len(train_loader.dataset), int(100.*batch_idx/len(train_loader)),\
                            loss.item(), accuracy.item(), precision.item(), recall.item(), f1_score.item()), 
                    file=open(model_dir + '/' + 'output_batch.log','a'))

        if epoch % 1 == 0:
            model.eval()
            print("Epoch: {}".format(epoch))
            print(("[Train]       Loss: {:.2f}    Accuracy: {:.2f}    Precision: {:.2f}    Recall: {:.2f}    F1_score: {:.2f}"\
                + '').format(total_loss / t, total_accuracy / t, total_precision / t, total_recall / t, total_f1_score / t))

            # Save to log
            print(("Epoch: {}".format(epoch)), file=open(model_dir + '/' + 'output_epoch.log','a'))
            print(("[Train]       Loss: {:.2f}    Accuracy: {:.2f}    Precision: {:.2f}    Recall: {:.2f}    F1_score: {:.2f}"\
                + '').format(total_loss / t, total_accuracy / t, total_precision / t, total_recall / t, total_f1_score / t),
                file=open(model_dir + '/' + 'output_epoch.log','a'))
            
            total_loss, total_accuracy, total_precision, total_recall, total_f1_score = (0, 0, 0, 0, 0)
            
            for batch_idx, (lengths, x, y) in tqdm(enumerate(valid_loader)):

                if cuda:
                    x, y, lengths = x.to(device, non_blocking=non_blocking),\
                                        y.long().to(device, non_blocking=non_blocking),\
                                            lengths.to(device, non_blocking=non_blocking)

                # # TF representation (PyTorch)
                # x = stft_pytorch(x,
                #         fs=fs,
                #         wlen_sec=wlen_sec,
                #         win=window, 
                #         hop_percent=hop_percent,
                #         center=center,
                #         pad_mode=pad_mode,
                #         pad_at_end=pad_at_end) # shape = (freq_bins, frames)

                # # Power spectrogram
                # x = x[...,0]**2 + x[...,1]**2

                # # Apply log
                # x = torch.log(x + eps)

                # # Swap x_dim and seq_length axes
                # x = x.transpose(1,-1)
                
                # Normalize power spectrogram
                if std_norm:
                    x_norm = x - mean.T
                    x_norm /= (std + eps).T

                    y_hat_soft = model(x_norm, lengths) 
                else:
                    y_hat_soft = model(x, lengths)
                
                # y_hat_soft = torch.squeeze(y_hat_soft)
                loss = 0.
                for (length, pred, target) in zip(lengths, y_hat_soft, y):
                    loss += binary_cross_entropy(pred[:length], target[:length], eps)
                # loss /= len(lengths)

                total_loss += loss.item()
                # _, y_hat_hard = torch.max(y_hat_soft.data, 1)
                y_hat_soft = torch.sigmoid(y_hat_soft.detach())
                y_hat_hard = (y_hat_soft > 0.5).int()
                # y_hat_hard = (y_hat_soft.detach() > 0.5).int()

                # exclude padding from F1-score
                y_hat_hard_batch, y_batch = [], []
                accuracy, precision, recall, f1_score = 0., 0., 0., 0.
                for (length, pred, target) in zip(lengths, y_hat_hard, y):
                    acc, prec, rec, f1 = f1_loss(y_hat_hard=torch.flatten(pred[:length]),
                                                                    y=torch.flatten(target[:length]),
                                                                    epsilon=eps)
                    accuracy += acc
                    precision += prec
                    recall += rec
                    f1_score += f1
                accuracy /= len(lengths)
                precision /= len(lengths)
                recall /= len(lengths)
                f1_score /= len(lengths)

                total_accuracy += accuracy
                total_precision += precision
                total_recall += recall
                total_f1_score += f1_score

            print(("[Validation]  Loss: {:.2f}    Accuracy: {:.2f}    Precision: {:.2f}    Recall: {:.2f}    F1_score: {:.2f}"\
                + '').format(total_loss / m, total_accuracy / m, total_precision / m, total_recall / m, total_f1_score / m))

            # Save to log
            print(("[Validation]  Loss: {:.2f}    Accuracy: {:.2f}    Precision: {:.2f}    Recall: {:.2f}    F1_score: {:.2f}"\
                + '').format(total_loss / m, total_accuracy / m, total_precision / m, total_recall / m, total_f1_score / m),
                file=open(model_dir + '/' + 'output_epoch.log','a'))

            # Save model
            # NB: if using DataParallel, save model as model.module.state_dict()
            if hasattr(model, 'module'):
                torch.save(model.module.state_dict(), model_dir + '/' + 'Video_Net_epoch_{:03d}_vloss_{:.2f}.pt'.format(
                    epoch, total_loss / m))
            else:
                torch.save(model.state_dict(), model_dir + '/' + 'Video_Net_epoch_{:03d}_vloss_{:.2f}.pt'.format(
                    epoch, total_loss / m))

if __name__ == '__main__':
    main()
