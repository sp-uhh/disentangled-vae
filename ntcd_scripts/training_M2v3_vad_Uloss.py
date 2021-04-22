import sys
sys.path.append('.')

import os
import torch
import pickle
import h5py as h5
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from packages.utils import count_parameters
from packages.data_handling import HDF5CleanSpectrogramLabeledFrames
from packages.models.models import DeepGenerativeModel_v3
from packages.models.utils import U_loss, binary_cross_entropy

##################################### SETTINGS #####################################################

# Dataset
# dataset_size = 'subset'
dataset_size = 'complete'
upsampled = True

dataset_name = 'ntcd_timit'

# Labels
labels = 'vad_labels'

# System 
cuda = torch.cuda.is_available()
cuda_device = "cuda:0"
device = torch.device(cuda_device if cuda else "cpu")
num_workers = 16
pin_memory = True
non_blocking = True
rdcc_nbytes = 1024**2*40  # The number of bytes to use for the chunk cache
                           # Default is 1 Mb
                           # Here we are using 400Mb of chunk_cache_mem here
rdcc_nslots = 1e4 # The number of slots in the cache's hash table
                  # Default is 521
                  # ideally 100 x number of chunks that can be fit in rdcc_nbytes
                  # (see https://docs.h5py.org/en/stable/high/file.html?highlight=rdcc#chunk-cache)

# Deep Generative Model
x_dim = 513 
y_dim = 1
z_dim = 16
h_dim = [128, 128]
std_norm = False
eps = 1e-8

# Classifier
alpha = -1000.

# Training
batch_size = 128
# learning_rate = 1e-3
learning_rate = 1e-4
log_interval = 250
start_epoch = 1
end_epoch = 500

# model_name = 'ntcd_M2v3_VAD_Uloss_nonorm_hdim_{:03d}_{:03d}_zdim_{:03d}_end_epoch_{:03d}'.format(h_dim[0], h_dim[1], z_dim, end_epoch)
# model_name = 'ntcd_M2v3_VAD_Uloss_alpha_0.0_nonorm_hdim_{:03d}_{:03d}_zdim_{:03d}_end_epoch_{:03d}'.format(h_dim[0], h_dim[1], z_dim, end_epoch)
# model_name = 'ntcd_M2v3_VAD_Uloss_alpha_0.0_hardlabel_nonorm_hdim_{:03d}_{:03d}_zdim_{:03d}_end_epoch_{:03d}'.format(h_dim[0], h_dim[1], z_dim, end_epoch)
# model_name = 'ntcd_M2v3_VAD_Uloss_alpha_10.0_hardlabel_nonorm_hdim_{:03d}_{:03d}_zdim_{:03d}_end_epoch_{:03d}'.format(h_dim[0], h_dim[1], z_dim, end_epoch)
# model_name = 'ntcd_M2v3_VAD_Uloss_alpha_-10.0_hardlabel_nonorm_hdim_{:03d}_{:03d}_zdim_{:03d}_end_epoch_{:03d}'.format(h_dim[0], h_dim[1], z_dim, end_epoch)
# model_name = 'ntcd_M2v3_VAD_Uloss_alpha_-100.0_hardlabel_nonorm_hdim_{:03d}_{:03d}_zdim_{:03d}_end_epoch_{:03d}'.format(h_dim[0], h_dim[1], z_dim, end_epoch)
# model_name = 'ntcd_M2v3_VAD_Uloss_alpha_20.0_hardlabel_nonorm_hdim_{:03d}_{:03d}_zdim_{:03d}_end_epoch_{:03d}'.format(h_dim[0], h_dim[1], z_dim, end_epoch)
# model_name = 'ntcd_M2v3_VAD_Uloss_alpha_20.0_ytrue_nonorm_hdim_{:03d}_{:03d}_zdim_{:03d}_end_epoch_{:03d}'.format(h_dim[0], h_dim[1], z_dim, end_epoch)
# model_name = 'ntcd_M2v3_VAD_Uloss_alpha_20.0_yhathard_nonorm_hdim_{:03d}_{:03d}_zdim_{:03d}_end_epoch_{:03d}'.format(h_dim[0], h_dim[1], z_dim, end_epoch)
# model_name = 'ntcd_M2v3_VAD_Lloss_alpha_20.0_yhathard_nonorm_hdim_{:03d}_{:03d}_zdim_{:03d}_end_epoch_{:03d}'.format(h_dim[0], h_dim[1], z_dim, end_epoch)
# model_name = 'ntcd_M2v3_VAD_Lloss_alpha_-10.0_yhathard_nonorm_hdim_{:03d}_{:03d}_zdim_{:03d}_end_epoch_{:03d}'.format(h_dim[0], h_dim[1], z_dim, end_epoch)
model_name = 'ntcd_M2v3_VAD_Lloss_alpha_-1000.0_yhathard_nonorm_hdim_{:03d}_{:03d}_zdim_{:03d}_end_epoch_{:03d}'.format(h_dim[0], h_dim[1], z_dim, end_epoch)

# Data directories
input_video_dir = os.path.join('data', dataset_size, 'processed/')
output_dataset_file = os.path.join(input_video_dir, dataset_name, 'Clean' + '_' + labels + '_upsampled.h5')

#####################################################################################################

print('Load data')
train_dataset = HDF5CleanSpectrogramLabeledFrames(input_video_dir=input_video_dir, dataset_name=dataset_name, dataset_type='train',
                                                  dataset_size=dataset_size, labels=labels, upsampled=upsampled,
                                                  rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots)
valid_dataset = HDF5CleanSpectrogramLabeledFrames(input_video_dir=input_video_dir, dataset_name=dataset_name, dataset_type='validation',
                                                  dataset_size=dataset_size, labels=labels, upsampled=upsampled,
                                                  rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=None, 
                        batch_sampler=None, num_workers=num_workers, pin_memory=pin_memory, 
                        drop_last=False, timeout=0, worker_init_fn=None)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, sampler=None, 
                        batch_sampler=None, num_workers=num_workers, pin_memory=pin_memory, 
                        drop_last=False, timeout=0, worker_init_fn=None)

print('- Number of training samples: {}'.format(len(train_dataset)))
print('- Number of validation samples: {}'.format(len(valid_dataset)))

print('- Number of training batches: {}'.format(len(train_loader)))
print('- Number of validation batches: {}'.format(len(valid_loader)))

def main():
    print('Create model')
    model = DeepGenerativeModel_v3([x_dim, y_dim, z_dim, h_dim])
    if cuda: model = model.to(device, non_blocking=non_blocking)

    # Create model folder
    model_dir = os.path.join('models', model_name)
    os.makedirs(model_dir, exist_ok=True)

    if std_norm:
        print('Load mean and std')
        with h5.File(output_dataset_file, 'r') as file:
            mean = file['X_train_mean'][:]
            std = file['X_train_std'][:]

        mean = torch.tensor(mean).to(device)
        std = torch.tensor(std).to(device)

    # Start log file
    file = open(model_dir + '/' +'output_batch.log','w') 
    file = open(model_dir + '/' +'output_epoch.log','w') 

    # Optimizer settings
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    t = len(train_loader)
    m = len(valid_loader)
    print('- Number of learnable parameters: {}'.format(count_parameters(model)))

    print('Start training')
    for epoch in range(start_epoch, end_epoch):
        model.train()
        total_loss, total_U, total_likelihood, total_kl, total_classif = (0, 0, 0, 0, 0)
        for batch_idx, (x, y) in tqdm(enumerate(train_loader)):
            if cuda:
                x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)

            # # Enumerate choices of label
            # y0 = torch.zeros((x.size(0), y_dim)).to(device)
            # y1 = torch.ones((x.size(0), y_dim)).to(device)
            # y01 = torch.cat([y0,y1], dim=0)
            # x = x.repeat(len([y0,y1]), 1)
            # y = y.repeat(len([y0,y1]), 1)

            y_hat_soft = model.classify(x)

            # r, mu, logvar = model(x, y_hat_soft)

            y_hat_hard = (y_hat_soft > 0.5)
            r, mu, logvar = model(x, y_hat_hard)
            
            # r, mu, logvar = model(x, y)

            U, L, recon_loss, KL = U_loss(x, r, mu, logvar, y_hat_soft, eps)
            
            # Add - alpha * BCE
            classif_loss = alpha * binary_cross_entropy(y, y_hat_soft, eps)
            # loss = U - classif_loss
            loss = L - classif_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            total_U += U.item()
            total_likelihood += recon_loss.item()
            total_kl += KL.item()
            total_classif += classif_loss.item()

            # Save to log
            if batch_idx % log_interval == 0:
                print(('Train Epoch: {:2d}   [{:7d}/{:7d} ({:2d}%)]    '\
                    'Total: {:.10f}    U: {:.10f}    Recon.: {:.3f}    KL: {:.3f}    Classif.: {:.3f}    '\
                    + '').format(epoch, batch_idx*len(x), len(train_loader.dataset), int(100.*batch_idx/len(train_loader)),\
                            loss.item(), U.item(), recon_loss.item(), KL.item(), classif_loss.item()), 
                    file=open(model_dir + '/' + 'output_batch.log','a'))

        if epoch % 1 == 0:
            model.eval()

            print("Epoch: {}".format(epoch))
            print("[Train]\t\t Total: {:.3f}, U: {:.3f}, Recon.: {:.3f}, KL: {:.3f}, Classif.: {:.3f}"\
                "".format(total_loss / t, total_U / t, total_likelihood / t, total_kl / t, total_classif / t))

            # Save to log
            print(("Epoch: {}".format(epoch)), file=open(model_dir + '/' + 'output_epoch.log','a'))
            print(("[Train]\t\t Total: {:.3f}, U: {:.3f}, Recon.: {:.3f}, KL: {:.3f}, Classif.: {:.3f}"\
                "".format(total_loss / t, total_U / t, total_likelihood / t, total_kl / t, total_classif / t)),
                file=open(model_dir + '/' + 'output_epoch.log','a'))

            total_loss, total_U, total_likelihood, total_kl, total_classif = (0, 0, 0, 0, 0)

            for batch_idx, (x, y) in tqdm(enumerate(valid_loader)):

                if cuda:
                    x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)

                # # Enumerate choices of label
                # y0 = torch.zeros((x.size(0), y_dim)).to(device)
                # y1 = torch.ones((x.size(0), y_dim)).to(device)
                # y01 = torch.cat([y0,y1], dim=0)
                # x = x.repeat(len([y0,y1]), 1)
                # y = y.repeat(len([y0,y1]), 1)

                y_hat_soft = model.classify(x)
                
                # r, mu, logvar = model(x, y_hat_soft)

                y_hat_hard = (y_hat_soft > 0.5)
                r, mu, logvar = model(x, y_hat_hard)
                
                # r, mu, logvar = model(x, y)

                U, L, recon_loss, KL = U_loss(x, r, mu, logvar, y_hat_soft, eps)

                # Add - alpha * BCE
                classif_loss = alpha * binary_cross_entropy(y, y_hat_soft, eps)
                # loss = U - classif_loss
                loss = L - classif_loss

                total_loss += loss.item()
                total_U += U.item()
                total_likelihood += recon_loss.item()
                total_kl += KL.item()
                total_classif += classif_loss.item()
  
            print("[Validation]\t Total: {:.3f}, U: {:.3f}, Recon.: {:.3f}, KL: {:.3f}, Classif.: {:.3f}"\
                "".format(total_loss / m, total_U / m, total_likelihood / m, total_kl / m, total_classif / m))

            print(("[Validation]\t Total: {:.3f}, U: {:.3f}, Recon.: {:.3f}, KL: {:.3f}, Classif.: {:.3f}"\
                "".format(total_loss / m, total_U / m, total_likelihood / m, total_kl / m, total_classif / m)),
                file=open(model_dir + '/' + 'output_epoch.log','a'))

            # Save model
            torch.save(model.state_dict(), model_dir + '/' + 'M2_epoch_{:03d}_vloss_{:.2f}.pt'.format(
                epoch, total_loss / m))
            
if __name__ == '__main__':
    main()