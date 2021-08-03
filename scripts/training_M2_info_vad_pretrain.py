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
from packages.models.models import DeepGenerativeModel_v5
from packages.models.utils import elbo, binary_cross_entropy, binary_cross_entropy_v2, binary_cross_entropy_v3

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
alpha = 0.
beta = 10.

# Training
batch_size = 128
# learning_rate = 1e-3
learning_rate = 1e-4
log_interval = 250
start_epoch = 1
end_epoch = 500

model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_0.0_beta_10.0_pretrain_yhatsoft_nonorm_hdim_{:03d}_{:03d}_zdim_{:03d}_end_epoch_{:03d}'.format(h_dim[0], h_dim[1], z_dim, end_epoch)

pretrained_model_name = 'ntcd_M2_info_VAD_Lenc_aux_v3_alpha_10.0_beta_10.0_yhatsoft_nonorm_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_144_vloss_396.43'

# Data directories
input_video_dir = os.path.join('data', dataset_size, 'processed/')
output_dataset_file = os.path.join(input_video_dir, dataset_name, 'Clean' + '_' + labels + '_upsampled.h5')
pretrained_classif_dir = os.path.join('models', pretrained_model_name + '.pt')

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
    model = DeepGenerativeModel_v5([x_dim, y_dim, z_dim, h_dim])
    
    
    
    print('Load pretrained model')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_classif_dir)
    
    # 1. Load Resnet and filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'enc_dec_clf.classifier' in k}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # 3. load the new state dict
    model.load_state_dict(model_dict)



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
    # 1 optimizer per submodel
    optimizer_enc_dec_clf = torch.optim.Adam(model.enc_dec_clf.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    # optimizer_dec = torch.optim.Adam(model.decoder.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    optimizer_aux = torch.optim.Adam(model.auxiliary.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    print('Freeze classifier')
    for name, child in model.named_children():
        if name == 'enc_dec_clf.classifier':
            for param in child.parameters():
                param.requires_grad = False

    t = len(train_loader)
    m = len(valid_loader)
    print('- Number of learnable parameters: {}'.format(count_parameters(model)))

    print('Start training')
    for epoch in range(start_epoch, end_epoch):
        model.train()
        total_ELBO, total_kl, total_likelihood, total_enc, total_classif, total_aux = (0, 0, 0, 0, 0, 0)
        for batch_idx, (x, y) in tqdm(enumerate(train_loader)):
            if cuda:
                x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)

            # Compute all variables & losses
            # ELBO, rec_loss, KL
            y_hat_class_soft = model.classify_fromX(x)
            r, z, mu, logvar = model(x, y_hat_class_soft)
            # r, z, mu, logvar = model(x, y)
            ELBO, recon_loss, KL = elbo(x, r, mu, logvar, eps)

            # Add + alpha * BCE (Classifier)
            classif_loss = alpha * binary_cross_entropy(y_hat_class_soft, y, eps)
            
            # Add - beta * BCE (Encoder)
            # Train the encoder to NOT predict y from z
            y_hat_aux_soft = model.classify_fromZ(z)
            # aux_enc_loss = beta * binary_cross_entropy(y_hat_aux_soft, y, eps)
            # aux_enc_loss = beta * binary_cross_entropy_v2(y_hat_aux_soft, eps)
            aux_enc_loss = beta * binary_cross_entropy_v3(y_hat_aux_soft, eps)
            
            # Encoder / Decoder loss
            # enc_loss = ELBO + classif_loss - aux_enc_loss
            # enc_loss = ELBO + classif_loss + aux_enc_loss
            enc_loss = ELBO + classif_loss - aux_enc_loss

            # Add + beta * BCE (Aux)
            # Train the aux net to predict y from z
            y_hat_aux_soft = model.classify_fromZ(z.detach()) #detach: to ONLY update the AUX net #the prediction here for GT being predY
            aux_loss = beta * binary_cross_entropy(y_hat_aux_soft, y, eps)

            # Gradient of each loss
			#zero the grads - otherwise they will be acculated
			#fill in grads and do updates:
            # recon_loss.backward()
            # optimizer_dec.step()
            # optimizer_dec.zero_grad()

            enc_loss.backward()
            optimizer_enc_dec_clf.step()
            optimizer_enc_dec_clf.zero_grad()

            aux_loss.backward()
            optimizer_aux.step()
            optimizer_aux.zero_grad()

            # Total losses 
            total_ELBO += ELBO.item()
            total_kl += KL.item()
            total_likelihood += recon_loss.item()
            total_enc += enc_loss.item()
            total_classif += classif_loss.item()
            total_aux += aux_loss.item()

            # Save to log
            if batch_idx % log_interval == 0:
                print(('Train Epoch: {:2d}   [{:7d}/{:7d} ({:2d}%)]    '\
                    'ELBO: {:.3f}    KL: {:.3f}    Recon.: {:.3f}    Enc.: {:.3f}    Classif.: {:.3f}    Aux.: {:.3f}    '\
                    + '').format(epoch, batch_idx*len(x), len(train_loader.dataset), int(100.*batch_idx/len(train_loader)),\
                            ELBO.item(), KL.item(), recon_loss.item(), enc_loss.item(), classif_loss.item(), aux_loss.item()), 
                    file=open(model_dir + '/' + 'output_batch.log','a'))

        if epoch % 1 == 0:
            model.eval()

            print("Epoch: {}".format(epoch))
            print("[Train]\t\t ELBO: {:.3f}, KL: {:.3f}, Recon.: {:.3f}, Enc.: {:.3f}, Classif.: {:.3f}, Aux.: {:.3f}"\
                "".format(total_ELBO / t, total_kl / t, total_likelihood / t, total_enc / t, total_classif / t, total_aux / t))

            # Save to log
            print(("Epoch: {}".format(epoch)), file=open(model_dir + '/' + 'output_epoch.log','a'))
            print(("[Train]\t\t ELBO: {:.3f}, KL: {:.3f}, Recon.: {:.3f}, Enc.: {:.3f}, Classif.: {:.3f}, Aux.: {:.3f}"\
                "".format(total_ELBO / t, total_kl / t, total_likelihood / t, total_enc / t, total_classif / t, total_aux / t)),
                file=open(model_dir + '/' + 'output_epoch.log','a'))

            total_ELBO, total_kl, total_likelihood, total_enc, total_classif, total_aux = (0, 0, 0, 0, 0, 0)

            for batch_idx, (x, y) in tqdm(enumerate(valid_loader)):

                if cuda:
                    x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)

                # Compute all variables & losses
                # ELBO, rec_loss, KL
                y_hat_class_soft = model.classify_fromX(x)
                r, z, mu, logvar = model(x, y_hat_class_soft)
                # r, z, mu, logvar = model(x, y)
                ELBO, recon_loss, KL = elbo(x, r, mu, logvar, eps)

                # Add + alpha * BCE (Classifier)
                classif_loss = alpha * binary_cross_entropy(y_hat_class_soft, y, eps)
                
                # Add - beta * BCE (Encoder)
                # Train the encoder to NOT predict y from z
                y_hat_aux_soft = model.classify_fromZ(z)
                # aux_enc_loss = beta * binary_cross_entropy(y_hat_aux_soft, y, eps)
                # aux_enc_loss = beta * binary_cross_entropy_v2(y_hat_aux_soft, eps)
                aux_enc_loss = beta * binary_cross_entropy_v3(y_hat_aux_soft, eps)
                
                # Encoder / Decoder loss
                # enc_loss = ELBO + classif_loss - aux_enc_loss
                # enc_loss = ELBO + classif_loss + aux_enc_loss
                enc_loss = ELBO + classif_loss - aux_enc_loss

                # Add + beta * BCE (Aux)
                # Train the aux net to predict y from z
                y_hat_aux_soft = model.classify_fromZ(z.detach()) #detach: to ONLY update the AUX net #the prediction here for GT being predY
                aux_loss = beta * binary_cross_entropy(y_hat_aux_soft, y, eps)

                # Total losses
                total_ELBO += ELBO.item()
                total_kl += KL.item()
                total_likelihood += recon_loss.item()
                total_enc += enc_loss.item()
                total_classif += classif_loss.item()
                total_aux += aux_loss.item()
  
            print("[Validation]\t ELBO: {:.3f}, KL: {:.3f}, Recon.: {:.3f}, Enc.: {:.3f}, Classif.: {:.3f}, Aux.: {:.3f}"\
                "".format(total_ELBO / m, total_kl / m, total_likelihood / m, total_enc / m, total_classif / m, total_aux / m))

            print(("[Validation]\t ELBO: {:.3f}, KL: {:.3f}, Recon.: {:.3f}, Enc.: {:.3f}, Classif.: {:.3f}, Aux.: {:.3f}"\
                "".format(total_ELBO / m, total_kl / m, total_likelihood / m, total_enc / m, total_classif / m, total_aux / m)),
                file=open(model_dir + '/' + 'output_epoch.log','a'))

            # Save model
            torch.save(model.state_dict(), model_dir + '/' + 'M2_epoch_{:03d}_vloss_{:.2f}.pt'.format(
                epoch, total_enc / m))
            
if __name__ == '__main__':
    main()