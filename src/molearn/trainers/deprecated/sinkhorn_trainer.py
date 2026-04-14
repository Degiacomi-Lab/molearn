import os 
import glob 
import numpy as np 
import torch
from molearn.loss_functions import openmm_energy
from molearn.data import PDBData
import json
from time import time
try:
    from geomloss import SamplesLoss
except ImportError as e:
    import warnings
    warnings.warn(f'{e}. Will not be able to use sinkhorn because geomloss is not installed.')

import shutil
from copy import deepcopy


class TrainingFailure(Exception):
    pass


class Sinkhorn_Trainer():

    def __init__(self, device=None, latent_dim=2, log_filename='default_log_file.dat'):
        if not device:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device
        print(f'device: {self.device}')
        self.best = None
        self.best_checkpoint_filename = None
        self.step = 0
        self.verbose = True
        self.log_filename = log_filename
        self.latent_dim = latent_dim
        self.sinkhorn = SamplesLoss(loss='sinkhorn', p=2, blur=0.05)
        self.save_time = time()

    def prepare_physics(self, physics_scaling_factor=0.1, clamp_threshold=10000, clamp=False, start_physics_at=0, **kwargs):
        self.start_physics_at = start_physics_at
        self.psf = physics_scaling_factor
        if clamp:
            clamp_kwargs = dict(max=clamp_threshold, min=-clamp_threshold)
        else:
            clamp_kwargs = None
        self.physics_loss = openmm_energy(self.mol, self.std, clamp=clamp_kwargs, platform='CUDA' if self.device == torch.device('cuda') else 'Reference', atoms=self._data.atoms, **kwargs)
        
    def get_network_summary(self,):
        
        def get_parameters(trainable_only, model):
            return sum(p.numel() for p in model.parameters() if (p.requires_grad and trainable_only))

        return dict(
            decoder_trainable=get_parameters(True, self.decoder),
            decoder_total=get_parameters(False, self.decoder))

    def set_data(self, data,*args, **kwargs):
        if isinstance(data, PDBData):
            train_data, valid_data = data.get_datasets(*args, **kwargs)
            self.train_data, self.valid_data = train_data.to(self.device), valid_data.to(self.device)
        else:
            raise NotImplementedError('Have not implemented this method to use any data other than PDBData yet')
        self.std = data.std
        self.mean = data.mean
        self.mol = data.mol
        self._data = data

    def set_dataloader(self, data, *args, **kwargs):
        if isinstance(data, PDBData):
            train_dataloader, valid_dataloader = data.get_dataloader(*args, **kwargs)
            
            def cycle(iterable):
                while True:
                    for x in iterable:
                        yield x
                        
            self.train_iterator = iter(cycle(train_dataloader))
            self.valid_iterator = iter(cycle(valid_dataloader))
        else:
            raise NotImplementedError('Have not implemented this method to use any data other than PDBData yet')
        self.std = data.std
        self.mean = data.mean
        self.mol = data.mol
        self._data = data

    def get_adam_opt(self, *args, **kwargs):
        self.opt = torch.optim.AdamW(self.decoder.parameters(), *args, **kwargs)
    
    def log(self, log_dict, verbose=None):
        dump = json.dumps(log_dict)
        if verbose or self.verbose:
            print(dump)
        with open(self.log_filename, 'a') as f:
            f.write(dump+'\n')

    def run(self, steps=100, validate_every=10, log_filename=None, checkpoint_frequency=1, checkpoint_folder='checkpoints', verbose=None):
        if log_filename is not None:
            self.log_filename = log_filename
        if verbose is not None:
            self.verbose = verbose
        start_step = self.step
        finish_step = start_step+steps
        number_of_validations = 0
        while self.step<finish_step:
            time1 = time()
            train_logs = self.training_n_steps(steps=validate_every)
            memory = torch.cuda.max_memory_allocated()/1000000.0
            time2 = time()
            valid_logs = self.validation_one_step()
            number_of_validations+=1
            time3 = time()
            if self.best is None or self.best > valid_logs['valid_loss']:
                self.checkpoint(valid_logs, checkpoint_folder)
            elif number_of_validations % checkpoint_frequency==0:
                self.checkpoint(valid_logs, checkpoint_folder)
            time4 = time()
            logs = {'step':self.step, **train_logs, **valid_logs,
                    'Memory': memory,
                    'train_seconds':time2-time1,
                    'valid_seconds':time3-time2,
                    'checkpoint_seconds': time4-time3,
                    'total_seconds':time4-time1}
            self.log(logs)
            if np.isnan(logs['valid_loss']) or np.isnan(logs['train_loss']):
                raise TrainingFailure('nan received, failing')

    def training_n_steps(self, steps):
        self.decoder.train()
        results = {}
        for i in range(steps):
            self.opt.zero_grad()
            train_result = self.train_step()
            train_result['loss'].backward()
            self.opt.step()
            self.step+=1
            if i == 0:
                results = {key:value.item() for key, value in train_result.items()}
            else:
                for key in train_result.keys():
                    results[key] += train_result[key].item()
        return {f'train_{key}': results[key]/steps for key in results.keys()}

    def validation_one_step(self):
        self.decoder.eval()
        result = self.valid_step()
        return {f'valid_{key}': result[key].item() for key in result.keys()}

    def train_step(self):
        data = self.train_data
        z = torch.randn(data.shape[0], self.latent_dim,1).to(self.device)
        structures = self.decoder(z)[:,:,:data.shape[2]]
        loss = self.sinkhorn(structures.reshape(structures.size(0), -1), data.reshape(data.size(0), -1))
        return dict(loss=loss)

    def valid_step(self):
        with torch.no_grad():
            data = self.valid_data
            z = torch.randn(data.shape[0], self.latent_dim).to(self.device)
            structures = self.decoder(z)[:,:,:data.shape[2]]
            loss = self.sinkhorn(structures.reshape(structures.size(0), -1), data.reshape(data.size(0), -1))
            energy = self.physics_loss(structures)
            energy[energy.isinf()] = 1e35
            energy = torch.clamp(energy, max=1e34)
            energy = energy.nanmean()

        z_0 = torch.zeros_like(z).requires_grad_()
        structures_0 = self.decoder(z_0)[:, :, :data.shape[2]]
        inner_loss = ((structures_0-data)**2).sum(1).mean()
        encoded = -torch.autograd.grad(inner_loss, [z_0], create_graph=True, retain_graph=True)[0]
        with torch.no_grad():
            structures_encoded = self.decoder(encoded)[:,:,:data.shape[2]]
        se = (structures_encoded-data)**2
        mse_loss = se.mean()
        rmsd = se.sum(1).mean().sqrt()*self.std
        if time()-self.save_time>120.:
            # coords = []
            with torch.no_grad():
                z1_index, z2_index = self.get_extrema()
                z1 = encoded[z1_index].unsqueeze(0)
                z2 = encoded[z2_index].unsqueeze(0)
                frames = 20
                ts = torch.linspace(0,1,frames).to(self.device).unsqueeze(-1)
                # from IPython import embed
                # embed(headre='valid')
                zinterp =(1-ts)*z1 + ts*z2
                if zinterp.shape == (2,frames):
                    zinterp = zinterp.permute(1,0)
                interp_structures = self.decoder(zinterp)[:, :, :data.shape[2]]
            mol = deepcopy(self.mol)
            mol.coordinates = (interp_structures.permute(0, 2, 1)*self.std).detach().cpu().numpy()
            mol.write_pdb('sample_interp.pdb', split_struc=False)
            self.save_time = time()

        return dict(loss=loss, physics_loss=energy,mse_loss=mse_loss, rmsd=rmsd)

    def update_optimiser_hyperparameters(self, **kwargs):
        for g in self.opt.param_groups:
            for key, value in kwargs.items():
                g[key] = value

    def checkpoint(self, valid_logs, checkpoint_folder, loss_key='valid_loss'):
        valid_loss = valid_logs[loss_key]
        if not os.path.exists(checkpoint_folder):
            os.mkdir(checkpoint_folder)
        torch.save({'step':self.step,
                    'model_state_dict': self.decoder.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict(),
                    'loss': valid_loss,
                    'network_kwargs': self._decoder_kwargs,
                    'atoms': self._data.atoms,
                    'std': self.std,
                    'mean': self.mean},
                   f'{checkpoint_folder}/last.ckpt')

        if self.best is None or self.best > valid_loss:
            filename = f'{checkpoint_folder}/checkpoint_step{self.step}_loss{valid_loss}.ckpt'
            shutil.copyfile(f'{checkpoint_folder}/last.ckpt', filename)
            if self.best is not None:
                os.remove(self.best_checkpoint_filename)
            self.best_checkpoint_filename = filename
            self.best_step = self.step
            self.best = valid_loss

    def load_checkpoint(self, checkpoint_name, checkpoint_folder, load_optimiser=True):
        if checkpoint_name=='best':
            if self.best_checkpoint_filename is not None:
                _name = self.best_checkpoint_filename
            else:
                ckpts = glob.glob(checkpoint_folder+'/checkpoint_step*')
                indexs = [x.rfind('loss') for x in ckpts]
                losses = [float(x[y+4:-5]) for x,y in zip(ckpts, indexs)]
                _name = ckpts[np.argmin(losses)]
        elif checkpoint_name =='last':
            _name = f'{checkpoint_folder}/last.ckpt'
        else:
            _name = f'{checkpoint_folder}/{checkpoint_name}'
        checkpoint = torch.load(_name)
        if not hasattr(self, 'decoder'):
            self.get_network(decoder_kwargs=checkpoint['network_kwargs'])

        self.decoder.load_state_dict(checkpoint['model_state_dict'])
        if load_optimiser:
            if not hasattr(self, 'opt'):
                self.get_adam_opt(dict(lr=1e-3))
            self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        self.step = step

    def get_extrema(self):
        # self.train_data [B, 3, N]
        if hasattr(self, '_extrema'):
            return self._extrema
        a = self.valid_data
        B = a.shape[0]
        with torch.no_grad():
            m = ((a.repeat_interleave(B,dim=0)-a.repeat(B, 1, 1))**2).sum(1).mean(-1).argmax()
        self._extrema = (m//B, m % B)
        return self._extrema
