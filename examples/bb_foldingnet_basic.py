import sys
import os
import shutil
import numpy as np
from time import time
import torch
import biobox as bb
import csv
from IPython import embed
sys.path.insert(0,'/home2/wppj21/Workshop/molearn/src')
import molearn
from torch import nn
import torch.nn.functional as F
import random
from molearn.scoring import Parallel_DOPE_Score
import argparse

def index_points(point_clouds, index):
    """
    Given a batch of tensor and index, select sub-tensor.
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, N, k]
    Return:
        new_points:, indexed points data, [B, N, k, C]
    """
    device = point_clouds.device
    batch_size = point_clouds.shape[0]
    view_shape = list(index.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(index.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = point_clouds[batch_indices, index, :]
    return new_points


def knn(x, k):
    """
    K nearest neighborhood.
    Parameters
    ----------
        x: a tensor with size of (B, C, N)
        k: the number of nearest neighborhoods
    
    Returns
    -------
        idx: indices of the k nearest neighborhoods with size of (B, N, k)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, 1, N), (B, N, N), (B, N, 1) -> (B, N, N)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (B, N, k)
    return idx


class GraphLayer(nn.Module):
    """
    Graph layer.
    in_channel: it depends on the input of this network.
    out_channel: given by ourselves.
    """
    def __init__(self, in_channel, out_channel, k=16):
        super(GraphLayer, self).__init__()
        self.k = k
        self.conv = nn.Conv1d(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        """
        Parameters
        ----------
            x: tensor with size of (B, C, N)
        """
        # KNN
        knn_idx = knn(x, k=self.k)  # (B, N, k)
        knn_x = index_points(x.permute(0, 2, 1), knn_idx)  # (B, N, k, C)

        # Local Max Pooling
        x = torch.max(knn_x, dim=2)[0].permute(0, 2, 1)  # (B, N, C)
        
        # Feature Map
        x = F.relu(self.bn(self.conv(x)))
        return x


class Encoder(nn.Module):
    """
    Graph based encoder.
    """
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv1d(12, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)

        self.graph_layer1 = GraphLayer(in_channel=64, out_channel=128, k=16)
        self.graph_layer2 = GraphLayer(in_channel=128, out_channel=1024, k=16)

        self.conv4 = nn.Conv1d(1024, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(512, 2,1)
    def forward(self, x):
        b, c, n = x.size()

        # get the covariances, reshape and concatenate with x
        knn_idx = knn(x, k=16)
        knn_x = index_points(x.permute(0, 2, 1), knn_idx)  # (B, N, 16, 3)
        mean = torch.mean(knn_x, dim=2, keepdim=True)
        knn_x = knn_x - mean
        covariances = torch.matmul(knn_x.transpose(2, 3), knn_x).view(b, n, -1).permute(0, 2, 1)
        x = torch.cat([x, covariances], dim=1)  # (B, 12, N)

        # three layer MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))


        # two consecutive graph layers
        x = self.graph_layer1(x)
        x = self.graph_layer2(x)

        x = self.bn4(self.conv4(x))

        x = torch.max(x, dim=-1)[0].unsqueeze(-1)
        
        x = self.conv5(x)
        return x


class FoldingLayer(nn.Module):
    """
    The folding operation of FoldingNet
    """

    def __init__(self, in_channel: int, out_channels: list):
        super(FoldingLayer, self).__init__()

        layers = []
        for oc in out_channels[:-1]:
            conv = nn.Conv1d(in_channel, oc, 3,1,1)
            bn = nn.BatchNorm1d(oc)
            active = nn.ReLU(inplace=True)
            layers.extend([conv, bn, active])
            in_channel = oc
        out_layer = nn.Conv1d(in_channel, out_channels[-1], 3,1,1)
        layers.append(out_layer)
        
        self.layers = nn.Sequential(*layers)

    def forward(self, *args):
        """
        Parameters
        ----------
            grids: reshaped 2D grids or intermediam reconstructed point clouds
        """
        # concatenate
        #try:
        #    x = torch.cat([*args], dim=1)
        #except:
        #    for arg in args:
        #        print(arg.shape)
        #    raise
        x = torch.cat([*args], dim=1)
        # shared mlp
        x = self.layers(x)
        
        return x

class Decoder_Layer(nn.Module):
    """
    Decoder Module of FoldingNet
    """

    def __init__(self, in_points, out_points, in_channel, out_channel,**kwargs):
        super(Decoder_Layer, self).__init__()

        # Sample the grids in 2D space
        #xx = np.linspace(-0.3, 0.3, 45, dtype=np.float32)
        #yy = np.linspace(-0.3, 0.3, 45, dtype=np.float32)
        #self.grid = np.meshgrid(xx, yy)   # (2, 45, 45)
        self.out_points = out_points
        self.grid = torch.linspace(-0.5, 0.5, out_points).view(1,-1)
        # reshape
        #self.grid = torch.Tensor(self.grid).view(2, -1)  # (2, 45, 45) -> (2, 45 * 45)
        assert out_points%in_points==0
        self.m = out_points//in_points

        self.fold1 = FoldingLayer(in_channel + 1, [512, 512, out_channel])
        self.fold2 = FoldingLayer(in_channel + out_channel+1, [512, 512, out_channel])

    def forward(self, x):
        """
        x: (B, C)
        """
        batch_size = x.shape[0]

        # repeat grid for batch operation
        grid = self.grid.to(x.device)                      # (2, 45 * 45)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 2, 45 * 45)
        
        # repeat codewords
        x = x.repeat_interleave(self.m, dim=-1)            # (B, 512, 45 * 45)
        
        # two folding operations
        recon1 = self.fold1(grid,x)
        recon2 = recon1+self.fold2(grid,x, recon1)
        
        return recon2
class Decoder(nn.Module):
    """
    Decoder Module of FoldingNet
    """

    def __init__(self, out_points, in_channel=2, **kwargs):
        super(Decoder, self).__init__()

        # Sample the grids in 2D space
        #xx = np.linspace(-0.3, 0.3, 45, dtype=np.float32)
        #yy = np.linspace(-0.3, 0.3, 45, dtype=np.float32)
        #self.grid = np.meshgrid(xx, yy)   # (2, 45, 45)
        start_out = (out_points//128) +1


        self.out_points = out_points

        self.layer1 = Decoder_Layer(1,           start_out,    in_channel,3*128)
        self.layer2 = Decoder_Layer(start_out,   start_out*8,  3*128,     3*16)
        self.layer3 = Decoder_Layer(start_out*8, start_out*32, 3*16,      3*4)
        self.layer4 = Decoder_Layer(start_out*32,start_out*128,3*4,       3)

    def forward(self, x):
        """
        x: (B, C)
        """
        x = x.view(-1, 2, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class AutoEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.encoder = Encoder(*args, **kwargs)
        self.decoder = Decoder(*args, **kwargs)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class OpenMM_log_Physics_Trainer(molearn.OpenMM_Physics_Trainer):
    def train_step(self, *args, **kwarps):
        raise NotImplementedError()

    def physics_train_step(self, batch):
        results = self.common_step(batch)
        results.update(self.common_physics_step(batch, self._internal['encoded']))

        with torch.no_grad():
            scale = (self.psf*results['mse_loss'])/(results['physics_loss'] +1e-5)
        results['loss'] = results['mse_loss']+scale*results['physics_loss']
        return results

    def disabled_train_step(self, batch):
        results = self.common_step(batch)
        results.update(self.common_physics_step(batch, self._internal['encoded']))
        results['loss'] = results['mse_loss']
        return results


    def prepare_physics(self, physics_scaling_factor=0.1, clamp_threshold = 10000, clamp=False, start_physics_at=0, remove_NB = False, **kwargs):
        self.start_physics_at = start_physics_at
        self.psf = physics_scaling_factor
        if clamp:
            clamp_kwargs = dict(max=clamp_threshold, min = -clamp_threshold)
        else:
            clamp_kwargs = None
        from molearn.loss_functions import openmm_energy
        self.physics_loss = openmm_energy(self.mol, self.std, clamp=clamp_kwargs, platform = 'CUDA' if self.device == torch.device('cuda') else 'Reference', atoms = self._data.atoms, remove_NB=remove_NB, **kwargs)
        self.physics_loss2 = openmm_energy(self.mol, self.std, clamp=clamp_kwargs, platform = 'CUDA' if self.device == torch.device('cuda') else 'Reference', atoms = self._data.atoms, remove_NB=not remove_NB, **kwargs)


    def valid_step(self, batch):
        results = self.common_step(batch)
        results.update(self.common_physics_step(batch, self._internal['encoded']))

        #rmsd
        rmsd = (((batch-self._internal['decoded'])*self.std)**2).sum(dim=1).mean().sqrt()
        results['RMSD'] = rmsd
        # second physics
        with torch.no_grad():
            energy = self.physics_loss2(self._internal['generated'])
            energy[energy.isinf()]=1e35
            energy = torch.clamp(energy, max=1e34)
            energy = energy.nanmean()
        results['physics_loss2'] = energy

        if self.first_valid_step and self.epoch%5==0:
            #if not hasattr(self, 'DOPE_time'):
                
            #    self.DOPE_time = time()
            self.first_valid_step = False
            if not hasattr(self, 'dope_score_class'):
                self.dope_score_class = Parallel_DOPE_Score(self.mol,processes=torch.get_num_threads())
            self.dope_scores = []
            pbatch = (self._internal['decoded'].permute(0,2,1)*self.std).data.cpu().numpy()
            for f in pbatch:
                if np.isfinite(f).all():
                    self.dope_scores.append(self.dope_score_class.get_score(f,refine=True))
            self.interp_dope_scores = []

            ipbatch = (self._internal['generated'].permute(0,2,1)*self.std).data.cpu().numpy()
            for f in ipbatch:
                if np.isfinite(f).all():
                    self.interp_dope_scores.append(self.dope_score_class.get_score(f,refine=True))

        results['loss'] = torch.log(results['mse_loss'])+self.psf*torch.log(results['physics_loss'])
        return results

    def valid_epoch(self, *args, **kwargs):
        self.first_valid_step = True
        memory = torch.cuda.max_memory_allocated()/1000000.0
        results = super().valid_epoch(*args, **kwargs)
        results['Memory'] = memory
        if self.epoch%5==0:
            t1 = time()
            dope = np.array([r.get() for r in self.dope_scores])
            idope = np.array([r.get() for r in self.interp_dope_scores])
            results['valid_DOPE'] = dope[:,0].mean()
            results['valid_DOPE_refined'] = dope[:,1].mean()
            results['valid_DOPE_interp'] = idope[:,0].mean()
            results['valid_DOPE_interp_refined'] = idope[:,1].mean()
            results['valid_DOPE_time'] = time()-t1
        results['lr'] = self._lr
        return results

    def checkpoint(self, epoch, valid_logs, checkpoint_folder, loss_key='valid_loss'):
        self.get_repeat(checkpoint_folder)

        valid_loss = valid_logs[loss_key]
        torch.save({'epoch':epoch,
                    'model_state_dict': self.autoencoder.state_dict(),
                    'optimizer_state_dict': self.optimiser.state_dict(),
                    'loss': valid_loss,
                    'network_kwargs': self._autoencoder_kwargs,
                    'atoms': self._data.atoms,
                    'std': self.std,
                    'mean': self.mean},
                   f'{checkpoint_folder}/last_{self._repeat}.ckpt')

        if self.best is None or self.best > valid_loss:
            filename = f'{checkpoint_folder}/checkpoint_{self._repeat}_epoch{epoch}_loss{valid_loss}.ckpt'
            shutil.copyfile(f'{checkpoint_folder}/last_{self._repeat}.ckpt', filename)
            if self.best is not None:
                os.remove(self.best_name)
            self.best_name = filename
            self.best_epoch = epoch
            self.best = valid_loss

    def get_repeat(self, checkpoint_folder):
        if not os.path.exists(checkpoint_folder):
            os.mkdir(checkpoint_folder)
        if not hasattr(self, '_repeat'):
            self._repeat = 0
            for i in range(1000):
                if not os.path.exists(checkpoint_folder+f'/last_{self._repeat}.ckpt'):
                    break#os.mkdir(checkpoint_folder)
                else:
                    self._repeat+=1
            else:
                raise Exception('Something went wrong, you surely havnt done 1000 repeats?')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", help = "repeat number", default = 0)
    parser.add_argument("--physics", help = 'How would you like to handle physics', choices = ['remove_NB', 'all', 'disabled', 'no_implicit'])
    return parser.parse_args()


if __name__ == '__main__':


    args = parse_args()
    print(args)

    data = molearn.PDBData()
    #data.import_pdb('/home2/projects/cgw/proteins/synaptotagmin/trajectory.pdb')
    data.import_pdb('/home2/projects/cgw/proteins/molearn/MurD_closed_open.pdb')
    #data.import_pdb('/home2/projects/cgw/proteins/molearn/aggregated_trajs.pdb')
    data.fix_terminal()
    data.atomselect(atoms = ['CA', 'C', 'N', 'CB', 'O'])
    #data.atomselect(atoms = 'no_hydrogen')#['CA', 'C', 'N', 'CB', 'O'])
    trainer = OpenMM_log_Physics_Trainer(device=torch.device('cuda'))
    batch_size = 16
    trainer.set_data(data, batch_size=batch_size, validation_split=0.1, manual_seed = 25)
    trainer.use_physics = True
    if args.physics == 'remove_NB':
        trainer.prepare_physics(remove_NB = True)
        trainer.train_step = trainer.physics_train_step
    elif args.physics =='all':
        trainer.prepare_physics(remove_NB = False)
        trainer.train_step = trainer.physics_train_step
    elif args.physics == 'disabled':
        trainer.prepare_physics(remove_NB = True)
        trainer.train_step = trainer.disabled_train_step
    elif args.physics == 'no_implicit':
        trainer.prepare_physics(remove_NB = False, xml_file = ['amber14-all.xml',])
        trainer.train_step = trainer.physics_train_step
    
    network_kwargs =  {
        'out_points':  data.dataset.shape[-1],
        'in_channels':  2,
                    }
    trainer.autoencoder = AutoEncoder(**network_kwargs).to(trainer.device)
    trainer._autoencoder_kwargs = network_kwargs
    #trainer.autoencoder.decoder = nDecoder(2,3,2).to(trainer.device)

    #trainer.get_adam_optimiser(dict(lr=1e-3))#, momentum=0.9))
    trainer.optimiser = torch.optim.AdamW(trainer.autoencoder.parameters(), lr=1e-3, weight_decay = 0.0001)

    name = f'xbb_{args.physics}_r{args.repeat}'
    checkpoint_folder = f'{name}_checkpoints'
    log_folder = checkpoint_folder
    trainer.get_repeat(checkpoint_folder)
    log_filename = f'{trainer._repeat}_log_file.dat'

    runkwargs = dict(
        log_filename=log_filename,
        log_folder=log_folder,
        checkpoint_folder=checkpoint_folder,
        checkpoint_frequency=1)


    best = 1e24
    lrs = [1e-3, 1e-4, 1e-5]
    for lr in lrs:
        trainer.update_optimiser_hyperparameters(lr=lr)
        trainer._lr = lr
        while True:
            trainer.run(max_epochs = 32+trainer.epoch,**runkwargs)
            if not best>trainer.best:
                break
            best = trainer.best
    best = trainer.best
    fbest = trainer.best_name
    print('best ', best, 'best_name ', fbest)
