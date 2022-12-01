import sys
import os
import glob
import shutil
import numpy as np
import time
import torch
import biobox as bb
import csv
import molearn
from molearn.autoencoder import Autoencoder as Net
from molearn.loss_functions import Auto_potential
from molearn.pdb_data import PDBData
from molearn.openmm_loss import openmm_energy
import warnings
from decimal import Decimal

class TrainingFailure(Exception):
    pass

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
        self.extra_print_args = ''
        self.extra_write_args = ''
        self.scheduler = None

    def get_dataset(self, filename, batch_size=16, atoms="*", validation_split=0.1, pin_memory=True, dataset_sample_size=-1):
        '''
        :param filename: location of the pdb
        :param atoms: "*" for all atoms, ["CA", "C", "N", "CB", "O"]
        '''
        warnings.warn("deprecated class method", DeprecationWarning)
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

    def set_data(self, data, **kwargs):
        if isinstance(data, PDBData):
            self.set_dataloader(*data.get_dataloader(**kwargs))
        else:
            raise NotImplementedError('Have not implemented this method to use any data other than PDBData yet')
        self.std = data.std
        self.mean = data.mean
        self.mol = data.mol
        self._data = data


    def get_network(self, autoencoder_kwargs=None, max_number_of_atoms=None):
        self._autoencoder_kwargs = autoencoder_kwargs
        if isinstance(max_number_of_atoms, int):
            n_atoms = max_number_of_atoms
        else:
            n_atoms = self.mol.coordinates.shape[1]

        power = autoencoder_kwargs['depth']+2
        init_n = (n_atoms//(2**power))+1
        print(f'Given a number of atoms: {n_atoms}, init_n should be set to {init_n} '+
              f'allowing a maximum of {init_n*(2**power)} atoms')
        autoencoder_kwargs['init_n'] = init_n

        self.autoencoder = Net(**autoencoder_kwargs).to(self.device)

    def get_optimiser(self, optimiser_kwargs=None):
        self.optimiser = torch.optim.SGD(self.autoencoder.parameters(), **optimiser_kwargs)

    def get_adam_optimiser(self, optimiser_kwargs=None):
        self.optimiser = torch.optim.Adam(self.autoencoder.parameters(), **optimiser_kwargs)

    def set_reduceLROnPlateau(self, verbose=True):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, mode='min', patience=16, verbose=verbose)

    def run(self, max_epochs=1600, log_filename = 'log_file.dat', checkpoint_frequency=8, checkpoint_folder='checkpoints', allow_n_failures=10):
        #Not safe, might overide your stuff

        with open(log_filename, 'a') as fout:
            for attempt in range(allow_n_failures):
                try:
                    for epoch in range(self.epoch, max_epochs):
                        time1 = time.time()
                        self.epoch = epoch
                        train_loss = self.train_step(epoch)
                        time2 = time.time()
                        with torch.no_grad():
                            valid_loss = self.valid_step(epoch)
                        time3 = time.time()
                        if self.scheduler is not None:
                            self.scheduler.step(valid_loss)
                        if epoch%checkpoint_frequency==0:
                            self.checkpoint(epoch, valid_loss, checkpoint_folder)
                        time4 = time.time()
                        print('  '.join(['%.2E'%Decimal(s) if isinstance(s ,float) else str(s)  for s in ['epoch',epoch, 'tl', train_loss, 'vl', valid_loss, 'train(s)',time2-time1,'valid(s)', time3-time2, 'check(s)', time4-time3, self.extra_print_args]])+'\n')
                        fout.write('  '.join([str(s) for s in [epoch, train_loss, valid_loss, time2-time1, time3-time2, time4-time3, self.extra_write_args]])+'\n')
                        if np.isnan(valid_loss) or np.isnan(train_loss):
                            raise TrainingFailure('nan received, failing')
                except TrainingFailure:
                    if attempt==(allow_n_failures-1):
                        failure_message = f'Training Failure due to Nan in attempt {attempt}, end now/n'
                        print(failure_message)
                        fout.write(failure_message)
                        raise TrainingFailure('nan received, failing')
                    failure_message = f'Training Failure due to Nan in attempt {attempt}, try again from best/n'
                    print(failure_message)
                    fout.write(failure_message)
                    if hasattr(self, 'best'):
                        self.load_checkpoint('best', checkpoint_folder)
                        #self.epoch+=1
                else:
                    break


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

    def learning_rate_sweep(self, max_lr=100, min_lr=1e-5, number_of_iterations=1000, checkpoint_folder='checkpoint_sweep'):
        self.autoencoder.train()
        def cycle(iterable):
            while True:
                for i in iterable:
                    yield i
        init_loss = 0.0
        values = []
        data = iter(cycle(self.train_dataloader))
        for i in range(number_of_iterations):
            lr = min_lr*((max_lr/min_lr)**(i/number_of_iterations))
            self.update_optimiser_hyperparameters(lr=lr)
            batch = next(data)[0].to(self.device).float()

            self.optimiser.zero_grad()
            latent = self.autoencoder.encode(batch)
            output = self.autoencoder.decode(latent)[:,:,:batch.size(2)]
            mse_loss = ((batch-output)**2).mean()
            #print(f'{i} lr {lr} loss {mse_loss.item()}')

            mse_loss.backward()
            self.optimiser.step()
            values.append((lr,mse_loss.item()))
            if i ==0:
                self.checkpoint(self.epoch, mse_loss.item(), checkpoint_folder)
                init_loss = mse_loss.item()
            if mse_loss.item()>10*init_loss:
                break
        self.load_checkpoint('last', checkpoint_folder)
        values = np.array(values)
        print('min value ', values[values[:,1].argmin()])
        return values

    def update_optimiser_hyperparameters(self, **kwargs):
        for g in self.optimiser.param_groups:
            for key, value in kwargs.items():
                g[key] = value

    def checkpoint(self, epoch, valid_loss, checkpoint_folder):
        if not os.path.exists(checkpoint_folder):
            os.mkdir(checkpoint_folder)
        torch.save({'epoch':epoch,
                    'model_state_dict': self.autoencoder.state_dict(),
                    'optimizer_state_dict': self.optimiser.state_dict(),
                    'loss': valid_loss,
                    'network_kwargs': self._autoencoder_kwargs,
                    'atoms': self._data.atoms,
                    'std': self.std,
                    'mean': self.mean},
                   f'{checkpoint_folder}/last.ckpt')

        if self.best is None or self.best > valid_loss:
            filename = f'{checkpoint_folder}/checkpoint_epoch{epoch}_loss{valid_loss}.ckpt'
            shutil.copyfile(f'{checkpoint_folder}/last.ckpt', filename)
            if self.best is not None:
                os.remove(self.best_name)
            self.best_name = filename
            self.best = valid_loss

    def load_checkpoint(self, checkpoint_name, checkpoint_folder):
        if checkpoint_name=='best':
            if self.best_name is not None:
                _name = self.best_name
            else:
                ckpts = glob.glob(checkpoint_folder+'/checkpoint_epoch*')
                indexs = [x.rfind('loss') for x in ckpts]
                losses = [float(x[y+4:-5]) for x,y in zip(ckpts, indexs)]
                _name = ckpts[np.argmin(losses)]
        elif checkpoint_name =='last':
            _name = f'{checkpoint_folder}/last.ckpt'
        else:
            _name = f'{checkpoint_folder}/{checkpoint_name}'
        checkpoint = torch.load(_name)
        if not hasattr(self, 'autoencoder'):
            self.get_network(autoencoder_kwargs=checkpoint['network_kwargs'])

        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        self.epoch = epoch+1


class Molearn_Physics_Trainer(Molearn_Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_physics(self, physics_scaling_factor=0.1):
        self.psf = physics_scaling_factor
        self.physics_loss = Auto_potential(self._data.dataset[0]*self.std, pdb_atom_names = self._data.get_atominfo(), device = self.device, method = 'roll')

    def train_step(self, epoch):
        self.autoencoder.train()
        average_loss = 0.0
        average_mse = 0.0
        average_bond = 0.0
        average_angle = 0.0
        average_torsion = 0.0
        N = 0
        for i, batch in enumerate(self.train_dataloader):
            batch = batch[0].to(self.device)
            n = len(batch)
            self.optimiser.zero_grad()

            latent = self.autoencoder.encode(batch)
            alpha = torch.rand(int(n//2), 1, 1).type_as(latent)
            latent_interpolated = (1-alpha)*latent[:-1:2] + alpha*latent[1::2]

            generated = self.autoencoder.decode(latent_interpolated)[:,:,:batch.size(2)]
            bond, angle, torsion =  self.physics_loss._roll_bond_angle_torsion_loss(generated*self.std)
            output = self.autoencoder.decode(latent)[:,:,:batch.size(2)]
            mse_loss = ((batch-output)**2).mean()

            average_physics = (bond + angle + torsion)/n
            with torch.no_grad():
                scale = self.psf*mse_loss/average_physics
            final_loss = mse_loss+scale*average_physics

            final_loss.backward()
            self.optimiser.step()
            average_loss+=final_loss.item()*n
            average_mse += mse_loss.item()*n
            average_bond+=bond.item()
            average_angle+=angle.item()
            average_torsion+=torsion.item()
            N+=n
        self.extra_write_args =  '  '.join([str(s) for s in ['tm', average_mse/N, 'tb', average_bond/N, 'ta', average_angle/N, 'tt', average_torsion/N]])
        return average_loss/N

    def valid_step(self, epoch):
        self.autoencoder.eval()
        average_loss = 0.0
        average_mse = 0.0
        average_bond = 0.0
        average_angle = 0.0
        average_torsion = 0.0
        N = 0
        for batch in self.valid_dataloader:
            batch = batch[0].to(self.device)
            n = len(batch)

            latent = self.autoencoder.encode(batch)
            alpha = torch.rand(int(n//2), 1, 1).type_as(latent)
            latent_interpolated = (1-alpha)*latent[:-1:2] + alpha*latent[1::2]

            generated = self.autoencoder.decode(latent_interpolated)[:,:,:batch.size(2)]
            bond, angle, torsion =  self.physics_loss._roll_bond_angle_torsion_loss(generated*self.std)
            output = self.autoencoder.decode(latent)[:,:,:batch.size(2)]
            mse_loss = ((batch-output)**2).mean()

            average_physics = (bond + angle + torsion)/n
            scale = self.psf*mse_loss/average_physics

            final_loss = mse_loss+scale*average_physics
            average_loss+=final_loss.item()*n
            average_mse += mse_loss.item()*n
            average_bond+=bond.item()
            average_angle+=angle.item()
            average_torsion+=torsion.item()
            N+=n
        self.extra_write_args += '  '.join([str(s) for s in ['vm', average_mse/N, 'vb', average_bond/N, 'va', average_angle/N, 'vt', average_torsion/N]])
        return average_loss/N

class OpenMM_Physics_Trainer(Molearn_Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_physics(self, physics_scaling_factor=0.1, clamp_threshold = 10000, clamp=False, start_physics_at=0):
        self.start_physics_at = start_physics_at
        self.psf = physics_scaling_factor
        if clamp:
            clamp_kwargs = dict(max=clamp_threshold, min = -clamp_threshold)
        else:
            clamp_kwargs = None
        self.physics_loss = openmm_energy(self.mol, self.std, clamp=clamp_kwargs, platform = 'CUDA' if self.device == torch.device('cuda') else 'Reference', atoms = self._data.atoms)

    def train_step(self, epoch):
        self.autoencoder.train()
        average_loss = 0.0
        average_mse = 0.0
        average_openmm_energy = 0.0
        N = 0
        for i, batch in enumerate(self.train_dataloader):
            batch = batch[0].to(self.device)
            n = len(batch)
            self.optimiser.zero_grad()

            latent = self.autoencoder.encode(batch)
            alpha = torch.rand(int(n//2), 1, 1).type_as(latent)
            latent_interpolated = (1-alpha)*latent[:-1:2] + alpha*latent[1::2]

            generated = self.autoencoder.decode(latent_interpolated)[:,:,:batch.size(2)]
            energy = self.physics_loss(generated)
            output = self.autoencoder.decode(latent)[:,:,:batch.size(2)]
            mse_loss = ((batch-output)**2).mean()
            energy[energy.isinf()]=0.0
            average_physics = energy.nanmean()
            with torch.no_grad():
                scale = (self.psf*mse_loss/average_physics) +1.0e-37
            if epoch>=self.start_physics_at:
                final_loss = mse_loss+scale*average_physics
            else:
                final_loss = mse_loss

            final_loss.backward()
            self.optimiser.step()
            average_loss+=final_loss.item()*n
            average_mse += mse_loss.item()*n
            average_openmm_energy += average_physics.item()*n
            N+=n
        self.extra_write_args =  '  '.join([str(s) for s in ['tm', average_mse/N, 'to', average_openmm_energy/N]])
        return average_loss/N

    def valid_step(self, epoch):
        self.autoencoder.eval()
        average_loss = 0.0
        average_mse = 0.0
        average_openmm_energy = 0.0
        N = 0
        for batch in self.valid_dataloader:
            batch = batch[0].to(self.device)
            n = len(batch)

            latent = self.autoencoder.encode(batch)
            alpha = torch.rand(int(n//2), 1, 1).type_as(latent)
            latent_interpolated = (1-alpha)*latent[:-1:2] + alpha*latent[1::2]

            generated = self.autoencoder.decode(latent_interpolated)[:,:,:batch.size(2)]
            energy = self.physics_loss(generated)
            output = self.autoencoder.decode(latent)[:,:,:batch.size(2)]
            mse_loss = ((batch-output)**2).mean()

            energy[energy.isinf()]=0.0
            average_physics = energy.nanmean()
            scale = self.psf*mse_loss/average_physics

            if epoch>=self.start_physics_at:
                final_loss = mse_loss+scale*average_physics
            else:
                final_loss = mse_loss
            average_loss+=final_loss.item()*n
            average_mse += mse_loss.item()*n
            average_openmm_energy+=average_physics.item()*n
            N+=n
        self.extra_write_args += '  '.join([str(s) for s in ['vm', average_mse/N, 'vo', average_openmm_energy/N]])
        return average_loss/N



if __name__=='__main__':
    pass
