import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py as h5 # to read .mat files
from scipy.fftpack import idct
import math
import torch
import torchaudio
from packages.processing.stft import stft_pytorch

# Parameters
dataset_name = 'ntcd_timit'
if dataset_name == 'ntcd_timit':
    from packages.dataset.ntcd_timit import video_list, speech_list,\
        proc_noisy_clean_pair_dict, proc_video_audio_pair_dict

class HDF5SpectrogramLabeledFrames(Dataset):
    def __init__(self, output_h5_dir, dataset_type, rdcc_nbytes, rdcc_nslots):
        # Do not load hdf5 in __init__ if num_workers > 0
        self.output_h5_dir = output_h5_dir
        self.dataset_type = dataset_type
        self.rdcc_nbytes = rdcc_nbytes
        self.rdcc_nslots = rdcc_nslots
        with h5.File(self.output_h5_dir, 'r') as file:
            self.dataset_len = file["X_" + dataset_type].shape[-1]

    def open_hdf5(self):
        #We are using 400Mb of chunk_cache_mem here ("rdcc_nbytes" and "rdcc_nslots")
        self.f = h5.File(self.output_h5_dir, 'r', rdcc_nbytes=self.rdcc_nbytes, rdcc_nslots=self.rdcc_nslots)
        
        # Faster to open datasets once, rather than at every call of __getitem__
        self.data = self.f['X_' + self.dataset_type]
        self.labels = self.f['Y_' + self.dataset_type]

    def __getitem__(self, i):
        # Open hdf5 here if num_workers > 0
        if not hasattr(self, 'f'):
            self.open_hdf5()
        return self.data[:,i], self.labels[:,i]

    def __len__(self):
        return self.dataset_len

    def __del__(self): 
        if hasattr(self, 'f'):
            self.f.close()

class HDF5SequenceSpectrogramLabeledFrames(Dataset):
    def __init__(self, output_h5_dir, dataset_type, rdcc_nbytes, rdcc_nslots, seq_length):
        # Do not load hdf5 in __init__ if num_workers > 0
        self.output_h5_dir = output_h5_dir
        self.dataset_type = dataset_type
        self.rdcc_nbytes = rdcc_nbytes
        self.rdcc_nslots = rdcc_nslots
        self.seq_length = seq_length
        with h5.File(self.output_h5_dir, 'r') as file:
            self.dataset_len = file["X_" + dataset_type].shape[-1]

    def open_hdf5(self):
        #We are using 400Mb of chunk_cache_mem here ("rdcc_nbytes" and "rdcc_nslots")
        self.f = h5.File(self.output_h5_dir, 'r', rdcc_nbytes=self.rdcc_nbytes, rdcc_nslots=self.rdcc_nslots)
        
        # Faster to open datasets once, rather than at every call of __getitem__
        self.data = self.f['X_' + self.dataset_type]
        self.labels = self.f['Y_' + self.dataset_type]

    def __getitem__(self, i):
        # Open hdf5 here if num_workers > 0
        if not hasattr(self, 'f'):
            self.open_hdf5()
        
        # i = i * self.seq_length

        # data = torch.zeros(self.seq_length, 1)
        # target = torch.zeros(self.seq_length,1)

        # data[i] = torch.tensor(self.data[:,:,][0])
        # target[i] = torch.tensor(self.oudataframe.iloc[idx+i][1])

        # start_point = np.random.randint(matlab_frames_list_per_user.shape[0] - self.seq_length)
        # video_frames = video_frames[start_point:(start_point + self.seq_length)]
        #TODO: change and move to the next sequence without overlapping (or overlapping by half)
        if i < self.seq_length:
            # return smaller sequence
            data = np.array(self.data[...,:i+1])
            labels = np.array(self.labels[...,i:i+1])
            length = data.shape[-1]
            # return torch.Tensor(self.data[...,:i+1]), torch.Tensor(self.labels[...,i:i+1]) # Take only the last label
            return torch.Tensor(data), torch.Tensor(labels), length #, length # Take only the last label
        else:
            # return full sequence
            data = np.array(self.data[...,i+1-self.seq_length:i+1])
            labels = np.array(self.labels[...,i:i+1])
            length = data.shape[-1]
            # return torch.Tensor(self.data[...,i+1-self.seq_length:i+1]), torch.Tensor(self.labels[...,i:i+1]) # Take only the last label
            return torch.Tensor(data), torch.Tensor(labels), length #, length # Take only the last label

    def __len__(self):
        return self.dataset_len
        # return math.ceil(self.dataset_len / self.seq_length)

    def __del__(self): 
        if hasattr(self, 'f'):
            self.f.close()

class HDF5WholeSequenceSpectrogramLabeledFrames(Dataset):
    def __init__(self, output_h5_dir, dataset_type, rdcc_nbytes, rdcc_nslots, seq_length):
        # Do not load hdf5 in __init__ if num_workers > 0
        self.output_h5_dir = output_h5_dir
        self.dataset_type = dataset_type
        self.rdcc_nbytes = rdcc_nbytes
        self.rdcc_nslots = rdcc_nslots
        self.seq_length = seq_length
        with h5.File(self.output_h5_dir, 'r') as file:
            dataset_len = file["X_" + dataset_type].shape[-1]
            self.dataset_len = math.ceil(dataset_len / seq_length)

    def open_hdf5(self):
        #We are using 400Mb of chunk_cache_mem here ("rdcc_nbytes" and "rdcc_nslots")
        self.f = h5.File(self.output_h5_dir, 'r', rdcc_nbytes=self.rdcc_nbytes, rdcc_nslots=self.rdcc_nslots)
        
        # Faster to open datasets once, rather than at every call of __getitem__
        self.data = self.f['X_' + self.dataset_type]
        self.labels = self.f['Y_' + self.dataset_type]

    def __getitem__(self, i):
        # Open hdf5 here if num_workers > 0
        if not hasattr(self, 'f'):
            self.open_hdf5()
        
        i = i * self.seq_length

        # data = torch.zeros(self.seq_length, 1)
        # target = torch.zeros(self.seq_length,1)

        # data[i] = torch.tensor(self.data[:,:,][0])
        # target[i] = torch.tensor(self.oudataframe.iloc[idx+i][1])

        # start_point = np.random.randint(matlab_frames_list_per_user.shape[0] - self.seq_length)
        # video_frames = video_frames[start_point:(start_point + self.seq_length)]
        
        #TODO: change and move to the next sequence without overlapping (or overlapping by half)
        # return full sequence
        data = np.array(self.data[...,i:i+self.seq_length])
        labels = np.array(self.labels[...,i:i+self.seq_length])
        length = data.shape[-1]
        # return torch.Tensor(self.data[...,i+1-self.seq_length:i+1]), torch.Tensor(self.labels[...,i:i+1]) # Take only the last label
        return torch.Tensor(data), torch.Tensor(labels), length #, length # Take only the last label

    def __len__(self):
        return self.dataset_len
        # return math.ceil(self.dataset_len / self.seq_length)

    def __del__(self): 
        if hasattr(self, 'f'):
            self.f.close()

class WavWholeSequenceSpectrogramLabeledFrames(Dataset):
    def __init__(self, input_video_dir, dataset_type, labels='vad_labels', upsampled=False, dct=False, norm_video=False):
        # Do not load hdf5 in __init__ if num_workers > 0
        self.dataset_type = dataset_type        
        self.input_video_dir = input_video_dir
        self.labels = labels

        # Dict mapping video to clean speech
        self.video_file_paths, self.audio_file_paths = proc_video_audio_pair_dict(input_video_dir=input_video_dir,
                                                dataset_type=dataset_type,
                                                labels=labels,
                                                upsampled=upsampled,
                                                dct=dct,
                                                norm_video=norm_video)
        
        self.dataset_len = len(self.video_file_paths) # total number of utterances

    def __getitem__(self, i):
        # Read video
        h5_file_path = self.input_video_dir + self.video_file_paths[i]

        # Open HDF5 file
        with h5.File(h5_file_path, 'r') as file:
            data = np.array(file["X"][:])

        # Sequence length
        length = data.shape[-1]

        # Read label
        h5_file_path = self.input_video_dir + self.audio_file_paths[i]

        with h5.File(h5_file_path, 'r') as file:
            label = np.array(file["Y"][:])
        
        return torch.Tensor(data), torch.Tensor(label), length #, length # Take only the last label

    def __len__(self):
        return self.dataset_len

class NoisyWavWholeSequenceSpectrogramLabeledFrames(Dataset):
    def __init__(self,
                 input_video_dir, dataset_type,
                 dataset_size, labels='vad_labels', upsampled=False,
                 fs=16000, wlen_sec=64e-3, win='hann', hop_percent=0.25,
                 center=True, pad_mode='reflect', pad_at_end=True, eps=1e-8):
        # Do not load hdf5 in __init__ if num_workers > 0
        self.input_video_dir = input_video_dir
        self.dataset_type = dataset_type
        self.dataset_size = dataset_size        
        self.labels = labels
        self.upsampled = upsampled

        # STFT parameters
        self.fs = fs
        self.wlen_sec = wlen_sec
        self.win = win
        self.hop_percent = hop_percent
        self.center = center
        self.pad_mode = pad_mode
        self.pad_at_end = pad_at_end
        self.eps = eps

        # Dict mapping noisy speech to clean speech
        self.noisy_clean_pair_paths = proc_noisy_clean_pair_dict(input_speech_dir=input_video_dir,
                                                dataset_type=dataset_type,
                                                dataset_size=dataset_size,
                                                labels=labels,
                                                upsampled=upsampled)

        # Convert dict to tuples
        self.noisy_clean_pair_paths = list(self.noisy_clean_pair_paths.items())

        # # TODO: correct audio / target alignment (paths not matching)
        # input_clean_file_paths, \
        #     output_clean_file_paths = speech_list(input_speech_dir='data/complete/raw/',
        #                         dataset_type=dataset_type)

        # self.noisy_clean_pair_paths = [(input_clean_file_path, output_clean_file_path)
        #                 for input_clean_file_path, output_clean_file_path\
        #                     in zip(input_clean_file_paths, output_clean_file_paths)]

        self.dataset_len = len(self.noisy_clean_pair_paths) # total number of utterances

    def __getitem__(self, i):
        # select utterance
        (proc_noisy_file_path, clean_file_path) = self.noisy_clean_pair_paths[i]
        
        # Read noisy audio
        noisy_speech, fs_noisy_speech = torchaudio.load(self.input_video_dir + proc_noisy_file_path)
        # noisy_speech, fs_noisy_speech = torchaudio.load(self.input_video_dir + clean_file_path)
        noisy_speech = noisy_speech[0] # 1channel

        # Normalize audio
        noisy_speech = noisy_speech / (torch.max(torch.abs(noisy_speech)))

        # TF representation (PyTorch)
        noisy_speech_tf = stft_pytorch(noisy_speech,
                fs=self.fs,
                wlen_sec=self.wlen_sec,
                win=self.win, 
                hop_percent=self.hop_percent,
                center=self.center,
                pad_mode=self.pad_mode,
                pad_at_end=self.pad_at_end) # shape = (freq_bins, frames)

        # Power spectrogram
        data = noisy_speech_tf[...,0]**2 + noisy_speech_tf[...,1]**2

        # Apply log
        data = torch.log(data + self.eps)
       
        # Read label
        output_h5_file = self.input_video_dir + clean_file_path
        # output_h5_file = self.input_video_dir + os.path.splitext(clean_file_path)[0] + '_' + self.labels + '.h5'
        # output_h5_file = self.input_video_dir + os.path.splitext(clean_file_path)[0] + '_' + self.labels + '_upsampled.h5'

        with h5.File(output_h5_file, 'r') as file:
            label = np.array(file["Y"][:])
            label = torch.Tensor(label)

        # Reduce frames of audio or label
        if label.shape[-1] < data.shape[-1]:
            data = data[...,:label.shape[-1]]
        if label.shape[-1] > data.shape[-1]:
            data = label[...,:data.shape[-1]]
        
        # Sequence length
        length = data.shape[-1]
        
        return data, label, length

    def __len__(self):
        return self.dataset_len

class NoisyWavWholeSequenceWavLabeledFrames(Dataset):
    def __init__(self,
                 input_video_dir, dataset_type,
                 dataset_size, labels='vad_labels',
                 fs=16000, wlen_sec=64e-3, win='hann', hop_percent=0.25,
                 center=True, pad_mode='reflect', pad_at_end=True, eps=1e-8):
        # Do not load hdf5 in __init__ if num_workers > 0
        self.input_video_dir = input_video_dir
        self.dataset_type = dataset_type
        self.dataset_size = dataset_size        
        self.labels = labels

        # STFT parameters
        self.fs = fs
        self.wlen_sec = wlen_sec
        self.win = win
        self.hop_percent = hop_percent
        self.center = center
        self.pad_mode = pad_mode
        self.pad_at_end = pad_at_end
        self.eps = eps

        # Dict mapping noisy speech to clean speech
        self.noisy_clean_pair_paths = proc_noisy_clean_pair_dict(input_speech_dir=input_video_dir,
                                                dataset_type=dataset_type,
                                                dataset_size=dataset_size,
                                                labels=labels)

        # Convert dict to tuples
        self.noisy_clean_pair_paths = list(self.noisy_clean_pair_paths.items())

        self.dataset_len = len(self.noisy_clean_pair_paths) # total number of utterances

    def __getitem__(self, i):
        # select utterance
        (proc_noisy_file_path, clean_file_path) = self.noisy_clean_pair_paths[i]
        
        # Read noisy audio
        noisy_speech, fs_noisy_speech = torchaudio.load(self.input_video_dir + proc_noisy_file_path)
        # noisy_speech, fs_noisy_speech = torchaudio.load(self.input_video_dir + clean_file_path)
        noisy_speech = noisy_speech[0] # 1channel

        # Normalize audio
        data = noisy_speech / (torch.max(torch.abs(noisy_speech)))

        # Read label
        output_h5_file = self.input_video_dir + clean_file_path

        with h5.File(output_h5_file, 'r') as file:
            label = np.array(file["Y"][:])
            label = torch.Tensor(label)

        # Sequence length        
        time_length = data.shape[-1]
        tf_length = label.shape[-1]
        
        return data, label, time_length, tf_length

    def __len__(self):
        return self.dataset_len

class AudioVisualSequenceLabeledFrames(Dataset):
    def __init__(self,
                 input_video_dir, dataset_type,
                 dataset_size, labels='vad_labels', upsampled=False,
                 fs=16000, wlen_sec=64e-3, win='hann', hop_percent=0.25,
                 center=True, pad_mode='reflect', pad_at_end=True, eps=1e-8):
        # Do not load hdf5 in __init__ if num_workers > 0
        self.input_video_dir = input_video_dir
        self.dataset_type = dataset_type
        self.dataset_size = dataset_size        
        self.labels = labels
        self.upsampled = upsampled

        # STFT parameters
        self.fs = fs
        self.wlen_sec = wlen_sec
        self.win = win
        self.hop_percent = hop_percent
        self.center = center
        self.pad_mode = pad_mode
        self.pad_at_end = pad_at_end
        self.eps = eps

        # Dict mapping noisy speech to clean speech
        self.noisy_clean_pair_paths = proc_noisy_clean_pair_dict(input_speech_dir=input_video_dir,
                                                                 dataset_type=dataset_type,
                                                                 dataset_size=dataset_size,
                                                                 labels=labels,
                                                                 upsampled=upsampled)

        # Convert dict to tuples
        self.noisy_clean_pair_paths = list(self.noisy_clean_pair_paths.items())

        # # # TODO: correct audio / target alignment (paths not matching)
        # input_clean_file_paths, \
        #     output_clean_file_paths = speech_list(input_speech_dir='data/complete/raw/',
        #                         dataset_type=dataset_type)

        # self.noisy_clean_pair_paths = [(input_clean_file_path, output_clean_file_path)
        #                 for input_clean_file_path, output_clean_file_path\
        #                     in zip(input_clean_file_paths, output_clean_file_paths)]

        self.dataset_len = len(self.noisy_clean_pair_paths) # total number of utterances

    def __getitem__(self, i):
        # select utterance
        (proc_noisy_file_path, clean_file_path) = self.noisy_clean_pair_paths[i]
        
        # Read noisy audio
        noisy_speech, fs_noisy_speech = torchaudio.load(self.input_video_dir + proc_noisy_file_path)
        # noisy_speech, fs_noisy_speech = torchaudio.load(self.input_video_dir + clean_file_path)
        noisy_speech = noisy_speech[0] # 1channel

        # Normalize audio
        noisy_speech = noisy_speech / (torch.max(torch.abs(noisy_speech)))

        # TF representation (PyTorch)
        noisy_speech_tf = stft_pytorch(noisy_speech,
                fs=self.fs,
                wlen_sec=self.wlen_sec,
                win=self.win, 
                hop_percent=self.hop_percent,
                center=self.center,
                pad_mode=self.pad_mode,
                pad_at_end=self.pad_at_end) # shape = (freq_bins, frames)

        # Power spectrogram
        noisy_spectrogram = noisy_speech_tf[...,0]**2 + noisy_speech_tf[...,1]**2

        # Apply log
        noisy_spectrogram = torch.log(noisy_spectrogram + self.eps)
       
        # Read video
        output_h5_file = clean_file_path.replace('Clean', 'matlab_raw')
        output_h5_file = output_h5_file.replace('_' + self.labels, '')
        if self.upsampled:
            output_h5_file = os.path.splitext(output_h5_file)[0] + '.h5'
        else:
            output_h5_file = os.path.splitext(output_h5_file)[0] + '_normvideo.h5'
        output_h5_file = self.input_video_dir + output_h5_file

        # Open HDF5 file
        with h5.File(output_h5_file, 'r') as file:
            video = np.array(file["X"][:])
            video = torch.Tensor(video)

        # Read label
        output_h5_file = self.input_video_dir + clean_file_path
        # output_h5_file = self.input_video_dir + os.path.splitext(clean_file_path)[0] + '_' + self.labels + '.h5'
        # output_h5_file = self.input_video_dir + os.path.splitext(clean_file_path)[0] + '_' + self.labels + '_upsampled.h5'

        with h5.File(output_h5_file, 'r') as file:
            label = np.array(file["Y"][:])
            label = torch.Tensor(label)

        # Reduce frames of audio, video or label
        length = min(noisy_spectrogram.shape[-1], video.shape[-1], label.shape[-1])
        noisy_spectrogram = noisy_spectrogram[...,:length]
        video = video[...,:length]
        label = label[...,:length]

        # length = label.shape[-1]
        # time_length = noisy_speech.shape[-1]
                
        return noisy_spectrogram, video, label, length
        # return noisy_speech, video, label, length, time_length

    def __len__(self):
        return self.dataset_len

class AudioVisualSequenceWavLabeledFrames(Dataset):
    def __init__(self,
                 input_video_dir, dataset_type,
                 dataset_size, labels='vad_labels',
                 fs=16000, wlen_sec=64e-3, win='hann', hop_percent=0.25,
                 center=True, pad_mode='reflect', pad_at_end=True, eps=1e-8):
        # Do not load hdf5 in __init__ if num_workers > 0
        self.input_video_dir = input_video_dir
        self.dataset_type = dataset_type
        self.dataset_size = dataset_size        
        self.labels = labels

        # STFT parameters
        self.fs = fs
        self.wlen_sec = wlen_sec
        self.win = win
        self.hop_percent = hop_percent
        self.center = center
        self.pad_mode = pad_mode
        self.pad_at_end = pad_at_end
        self.eps = eps

        # Dict mapping noisy speech to clean speech
        self.noisy_clean_pair_paths = proc_noisy_clean_pair_dict(input_speech_dir=input_video_dir,
                                                                 dataset_type=dataset_type,
                                                                 dataset_size=dataset_size,
                                                                 labels=labels)

        # Convert dict to tuples
        self.noisy_clean_pair_paths = list(self.noisy_clean_pair_paths.items())

        self.dataset_len = len(self.noisy_clean_pair_paths) # total number of utterances

    def __getitem__(self, i):
        # select utterance
        (proc_noisy_file_path, clean_file_path) = self.noisy_clean_pair_paths[i]
        
        # Read noisy audio
        noisy_speech, fs_noisy_speech = torchaudio.load(self.input_video_dir + proc_noisy_file_path)
        # noisy_speech, fs_noisy_speech = torchaudio.load(self.input_video_dir + clean_file_path)
        noisy_speech = noisy_speech[0] # 1channel

        # Normalize audio
        data = noisy_speech / (torch.max(torch.abs(noisy_speech)))
       
        # Read video
        output_h5_file = clean_file_path.replace('Clean', 'matlab_raw')
        output_h5_file = output_h5_file.replace('_' + self.labels, '')
        output_h5_file = os.path.splitext(output_h5_file)[0] + '_upsampled.h5'
        output_h5_file = self.input_video_dir + output_h5_file

        # Open HDF5 file
        with h5.File(output_h5_file, 'r') as file:
            video = np.array(file["X"][:])
            video = torch.Tensor(video)

        # Read label
        output_h5_file = self.input_video_dir + clean_file_path

        with h5.File(output_h5_file, 'r') as file:
            label = np.array(file["Y"][:])
            label = torch.Tensor(label)

        # Reduce frames of audio, video or label
        time_length = data.shape[-1]
        tf_length = video.shape[-1]
                
        return data, video, label, time_length, tf_length

    def __len__(self):
        return self.dataset_len