import sys
import os
import shutil
import numpy as np
import time
import torch
import biobox as bb
import csv
#from IPython import embed
import molearn
from molearn.autoencoder import Autoencoder as Net


class Molearn_Trainer():
    def __init__(self, device = None, log_filename = 'log_file.dat'):
        if not device:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device
        print(f'device: {self.device}')
        self.best = None
        self.best_name = None
        self.epoch = 0

    def get_dataset(self, filename, batch_size=16, atoms="*", validation_split=0.1, pin_memory=True, dataset_sample_size=-1):
        '''
        :param filename: location of the pdb
        :param atoms: "*" for all atoms, ["CA", "C", "N", "CB", "O"]
        '''
        dataset, self.mean, self.std, self.atom_names, self.mol, test0, test1 = molearn.load_data(filename, atoms="*", dataset_sample_size=dataset_sample_size,
                device=torch.device('cpu'))
        print(f'Dataset.shape: {dataset.shape}')
        valid_size = int(len(dataset)*validation_split)
        train_size = len(dataset) - valid_size
        dataset = torch.utils.data.TensorDataset(dataset.float())
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
        self.valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True)

    def set_dataloader(self, train_dataloader=None, valid_dataloader=None):
        if train_dataloader is not None:
            self.train_dataloader = train_dataloader
        if valid_dataloader is not None:
            self.valid_dataloader = valid_dataloader

    def get_network(self, autoencoder_kwargs=None):
        self._autoencoder_kwargs = autoencoder_kwargs
        self.autoencoder = Net(**autoencoder_kwargs).to(self.device)

    def get_optimiser(self, optimiser_kwargs=None):
        self.optimiser = torch.optim.SGD(self.autoencoder.parameters(), **optimiser_kwargs)


    def run(self, max_epochs=1600, log_filename = 'log_file.dat', checkpoint_frequency=8, checkpoint_folder='checkpoints'):
        #Not safe, might overide your stuff

        with open(log_filename, 'a') as fout:
            for epoch in range(self.epoch, max_epochs):
                time1 = time.time()
                self.epoch = epoch
                train_loss = self.train_step(epoch)
                time2 = time.time()
                with torch.no_grad():
                    valid_loss = self.valid_step(epoch)
                time3 = time.time()
                if epoch%checkpoint_frequency==0:
                    self.checkpoint(epoch, valid_loss, checkpoint_folder)
                time4 = time.time()
                print(f'{epoch}\t{train_loss}\t{valid_loss}\t{time2-time1}\t{time3-time2}\t{time4-time3}\n')
                fout.write(f'{epoch}\t{train_loss}\t{valid_loss}\t{time2-time1}\t{time3-time2}\t{time4-time3}\n')

    def train_step(self,epoch):
        self.autoencoder.train()
        average_loss = 0.0
        N = 0
        for i, batch in enumerate(self.train_dataloader):
            batch = batch[0].to(self.device)
            self.optimiser.zero_grad()
            latent = self.autoencoder.encode(batch)
            output = self.autoencoder.decode(latent)[:,:,:batch.size(2)]
            mse_loss = ((batch-output)**2).mean()

            mse_loss.backward()
            self.optimiser.step()
            average_loss+=mse_loss.item()*len(batch)
            N+=len(batch)
        return average_loss/N

    def valid_step(self,epoch):
        self.autoencoder.eval()
        average_loss = 0.0
        N = 0
        for batch in self.valid_dataloader:
            batch = batch[0].to(self.device)
            latent = self.autoencoder.encode(batch)
            output = self.autoencoder.decode(latent)[:,:,:batch.size(2)]
            mse_loss = ((batch-output)**2).mean()
            average_loss+=mse_loss.item()*len(batch)
            N+=len(batch)
        return average_loss/N

    def checkpoint(self, epoch, valid_loss, checkpoint_folder):
        if not os.path.exists(checkpoint_folder):
            os.mkdir(checkpoint_folder)
        torch.save({'epoch':epoch,
                    'model_state_dict': self.autoencoder.state_dict(),
                    'optimizer_state_dict': self.optimiser.state_dict(),
                    'loss': valid_loss,
                    'network_kwargs': self._autoencoder_kwargs},
                f'{checkpoint_folder}/last.ckpt')

        if self.best is None or self.best > valid_loss:
            filename = f'{checkpoint_folder}/checkpoint_epoch{epoch}_loss{valid_loss}.ckpt'
            shutil.copyfile(f'{checkpoint_folder}/last.ckpt', filename)
            if self.best is not None:
                os.remove(self.best_name)
            self.best_name = filename
            self.best = valid_loss

if __name__=='__main__':
    pass
