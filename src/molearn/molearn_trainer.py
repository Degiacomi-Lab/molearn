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
from molearn.models.foldingnet import AutoEncoder as Net
from molearn.loss_functions import TorchProteinEnergy
from molearn.pdb_data import PDBData
from molearn.loss_functions import openmm_energy
import warnings
from decimal import Decimal
import json

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
        self.scheduler = None
        self.verbose = True
        self.log_filename = 'default_log_filename.json'
        self.scheduler_key = None

    def get_network_summary(self,):
        def get_parameters(trainable_only, model):
            return sum(p.numel() for p in model.parameters() if (p.requires_grad and trainable_only))

        return dict(
            encoder_trainable = get_parameters(True, self.autoencoder.encoder),
            encoder_total = get_parameters(False, self.autoencoder.encoder),
            decoder_trainable = get_parameters(True, self.autoencoder.decoder),
            decoder_total = get_parameters(False, self.autoencoder.decoder),
            autoencoder_trainable = get_parameters(True, self.autoencoder),
            autoencoder_total = get_parameters(False, self.autoencoder),
                      )


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


    def get_network(self, autoencoder_kwargs=None, max_number_of_atoms=None, network = None):
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
        net = network if network is not None else Net
        print('Network type' , type(net))
        self.autoencoder = net(**autoencoder_kwargs).to(self.device)

    def get_optimiser(self, optimiser_kwargs=None):
        self.optimiser = torch.optim.SGD(self.autoencoder.parameters(), **optimiser_kwargs)

    def get_adam_optimiser(self, optimiser_kwargs=None):
        self.optimiser = torch.optim.Adam(self.autoencoder.parameters(), **optimiser_kwargs)

    def set_reduceLROnPlateau(self, verbose=True,patience = 16):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, mode='min', patience=patience, verbose=verbose)
        #def override_step(self, logs):
        #    self.scheduler.step(logs['valid_loss'])
        #self.scheduler_step = override_step
        self.scheduler_key = 'valid_loss'

    def scheduler_step(self, logs):
        try:
            self.scheduler.step(logs.get(self.scheduler_key, None))
        except ValueError as e:
            print(e)



    def log(self, log_dict, verbose=None):
        dump = json.dumps(log_dict)
        if verbose or self.verbose:
            print(dump)
        with open(self.log_filename, 'a') as f:
            f.write(dump+'\n')


    def run(self, max_epochs=1600, log_filename = None, log_folder=None, checkpoint_frequency=8, checkpoint_folder='checkpoints', allow_n_failures=10, verbose=None):
        if log_filename is not None:
            self.log_filename = log_filename
            if log_folder is not None:
                if not os.path.exists(log_folder):
                    os.mkdir(log_folder)
                self.log_filename = log_folder+'/'+self.log_filename
        if verbose is not None:
            self.verbose = verbose

        for attempt in range(allow_n_failures):
            try:
                for epoch in range(self.epoch, max_epochs):
                    time1 = time.time()
                    train_logs = self.train_epoch(epoch)
                    time2 = time.time()
                    with torch.no_grad():
                        valid_logs = self.valid_epoch(epoch)
                    time3 = time.time()
                    if self.scheduler is not None:
                        self.scheduler_step(valid_logs)
                        train_logs['lr']= self.scheduler.get_last_lr()
                    if self.best is None or self.best > valid_logs['valid_loss']:
                        self.checkpoint(epoch, valid_logs, checkpoint_folder)
                    elif epoch%checkpoint_frequency==0:
                        self.checkpoint(epoch, valid_logs, checkpoint_folder)
                    time4 = time.time()
                    logs = {'epoch':epoch, **train_logs, **valid_logs,
                            'train_seconds':time2-time1,
                            'valid_seconds':time3-time2,
                            'checkpoint_seconds': time4-time3,
                            'total_seconds':time4-time1}
                    self.log(logs)
                    if np.isnan(logs['valid_loss']) or np.isnan(logs['train_loss']):
                        raise TrainingFailure('nan received, failing')
                    self.epoch+= 1
            except TrainingFailure:
                if attempt==(allow_n_failures-1):
                    failure_message = f'Training Failure due to Nan in attempt {attempt}, end now/n'
                    self.log({'Failure':failure_message})
                    raise TrainingFailure('nan received, failing')
                failure_message = f'Training Failure due to Nan in attempt {attempt}, try again from best/n'
                self.log({'Failure':failure_message})
                if hasattr(self, 'best'):
                    self.load_checkpoint('best', checkpoint_folder)
            else:
                break


    def train_epoch(self,epoch):
        self.autoencoder.train()
        N = 0
        results = {}
        for i, batch in enumerate(self.train_dataloader):
            batch = batch[0].to(self.device)
            self.optimiser.zero_grad()
            train_result = self.train_step(batch)
            train_result['loss'].backward()
            self.optimiser.step()
            if i == 0:
                results = {key:value.item()*len(batch) for key, value in train_result.items()}
            else:
                for key in train_result.keys():
                    results[key] += train_result[key].item()*len(batch)
            N+=len(batch)
        return {f'train_{key}': results[key]/N for key in results.keys()}

    def train_step(self, batch):
        results = self.common_step(batch)
        results['loss'] = results['mse_loss']
        return results

    def common_step(self, batch):
        self._internal = {}
        encoded = self.autoencoder.encode(batch)
        self._internal['encoded'] = encoded
        decoded = self.autoencoder.decode(encoded)[:,:,:batch.size(2)]
        self._internal['decoded'] = decoded
        return dict(mse_loss = ((batch-decoded)**2).mean())


    def valid_epoch(self,epoch):
        self.autoencoder.eval()
        N = 0
        results = {}
        for i, batch in enumerate(self.valid_dataloader):
            batch = batch[0].to(self.device)
            valid_result = self.valid_step(batch)
            if i == 0:
                results = {key:value.item()*len(batch) for key, value in valid_result.items()}
            else:
                for key in valid_result.keys():
                    results[key] += valid_result[key].item()*len(batch)
            N+=len(batch)
        return {f'valid_{key}': results[key]/N for key in results.keys()}

    def valid_step(self, batch):
        results = self.common_step(batch)
        results['loss'] = results['mse_loss']
        return results

    def learning_rate_sweep(self, max_lr=100, min_lr=1e-5, number_of_iterations=1000, checkpoint_folder='checkpoint_sweep',train_on='mse_loss', save=['loss', 'mse_loss']):
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
            result = self.train_step(batch)
            #result['loss']/=len(batch)
            result[train_on].backward()
            self.optimiser.step()
            values.append((lr,)+tuple((result[name].item() for name in save)))
            #print(i,lr, result['loss'].item())
            if i==0:
                init_loss = result[train_on].item()
            #if result[train_on].item()>1e6*init_loss:
            #    break
        values = np.array(values)
        print('min value ', values[np.nanargmin(values[:,1])])
        return values

    def update_optimiser_hyperparameters(self, **kwargs):
        for g in self.optimiser.param_groups:
            for key, value in kwargs.items():
                g[key] = value

    def checkpoint(self, epoch, valid_logs, checkpoint_folder, loss_key='valid_loss'):
        valid_loss = valid_logs[loss_key]
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
            self.best_epoch = epoch
            self.best = valid_loss

    def load_checkpoint(self, checkpoint_name, checkpoint_folder, load_optimiser=True):
        if checkpoint_name=='best':
            if self.best_name is not None:
                _name = self.best_name
            else:
                ckpts = glob.glob(checkpoint_folder+'/checkpoint_*')
                indexs = [x.rfind('loss') for x in ckpts]
                losses = [float(x[y+4:-5]) for x,y in zip(ckpts, indexs)]
                _name = ckpts[np.argmin(losses)]
        elif checkpoint_name =='last':
            _name = f'{checkpoint_folder}/last.ckpt'
        else:
            _name = f'{checkpoint_folder}/{checkpoint_name}'
        checkpoint = torch.load(_name)
        if not hasattr(self, 'autoencoder'):
            raise NotImplementedError('self.autoencoder does not exist, I have no way of knowing what network you want to load checkoint weights into yet, please set the network first')
            self.get_network(autoencoder_kwargs=checkpoint['network_kwargs'])

        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        if load_optimiser:
            if not hasattr(self, 'optimiser'):
                raise NotImplementedError('self.optimiser does not exist, I have no way of knowing what optimiser you previously used, please set it first.')
                self.get_optimiser(dict(lr=1e-20, momentum=0.9))
            self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        self.epoch = epoch+1


class Molearn_Physics_Trainer(Molearn_Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_physics(self, physics_scaling_factor=0.1):
        self.psf = physics_scaling_factor
        self.physics_loss = TorchProteinEnergy(self._data.dataset[0]*self.std, pdb_atom_names = self._data.get_atominfo(), device = self.device, method = 'roll')

    def common_physics_step(self, batch, latent, mse_loss):
        alpha = torch.rand(int(len(batch)//2), 1, 1).type_as(latent)
        latent_interpolated = (1-alpha)*latent[:-1:2] + alpha*latent[1::2]
        generated = self.autoencoder.decode(latent_interpolated)[:,:,:batch.size(2)]
        bond, angle, torsion =  self.physics_loss._roll_bond_angle_torsion_loss(generated*self.std)
        n = len(generated)
        bond/=n
        angle/=n
        torsion/=n
        _all = torch.tensor([bond, angle, torsion])
        _all[_all.isinf()]=1e35
        total_physics = _all.nansum()
        #total_physics = torch.nansum(torch.tensor([bond ,angle ,torsion]))

        return {'physics_loss':total_physics, 'bond_energy':bond, 'angle_energy':angle, 'torsion_energy':torsion}


    def train_step(self, batch):
        results = self.common_step(batch)
        results.update(self.common_physics_step(batch, self._internal['encoded'], results['mse_loss']))
        with torch.no_grad():
            scale = self.psf*results['mse_loss']/(results['physics_loss']+1e-5)
        final_loss = results['mse_loss']+scale*results['physics_loss']
        results['loss'] = final_loss
        return results

    def valid_step(self, batch):
        results = self.common_step(batch)
        results.update(self.common_physics_step(batch, self._internal['encoded'], results['mse_loss']))
        scale = self.psf*results['mse_loss']/(results['physics_loss']+1e-5)
        final_loss = results['mse_loss']+scale*results['physics_loss']
        results['loss'] = final_loss
        return results

class OpenMM_Physics_Trainer(Molearn_Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_physics(self, physics_scaling_factor=0.1, clamp_threshold = 10000, clamp=False, start_physics_at=0, **kwargs):
        self.start_physics_at = start_physics_at
        self.psf = physics_scaling_factor
        if clamp:
            clamp_kwargs = dict(max=clamp_threshold, min = -clamp_threshold)
        else:
            clamp_kwargs = None
        self.physics_loss = openmm_energy(self.mol, self.std, clamp=clamp_kwargs, platform = 'CUDA' if self.device == torch.device('cuda') else 'Reference', atoms = self._data.atoms, **kwargs)


    def common_physics_step(self, batch, latent):
        alpha = torch.rand(int(len(batch)//2), 1, 1).type_as(latent)
        latent_interpolated = (1-alpha)*latent[:-1:2] + alpha*latent[1::2]

        generated = self.autoencoder.decode(latent_interpolated)[:,:,:batch.size(2)]
        self._internal['generated'] = generated
        energy = self.physics_loss(generated)
        energy[energy.isinf()]=1e35
        energy = torch.clamp(energy, max=1e34)
        energy = energy.nanmean()

        return {'physics_loss':energy}#a if not energy.isinf() else torch.tensor(0.0)}

    def train_step(self, batch):
        results = self.common_step(batch)
        results.update(self.common_physics_step(batch, self._internal['encoded']))
        with torch.no_grad():
            scale = (self.psf*results['mse_loss'])/(results['physics_loss'] +1e-5)
        final_loss = results['mse_loss']+scale*results['physics_loss']
        results['loss'] = final_loss
        return results

    def valid_step(self, batch):
        results = self.common_step(batch)
        results.update(self.common_physics_step(batch, self._internal['encoded']))
        scale = (self.psf*results['mse_loss'])/(results['physics_loss'] +1e-5)
        final_loss = results['mse_loss']+scale*results['physics_loss']
        results['loss'] = final_loss
        return results




if __name__=='__main__':
    pass
