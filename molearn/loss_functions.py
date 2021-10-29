# Copyright (c) 2021 Venkata K. Ramaswamy, Samuel C. Musson, Chris G. Willcocks, Matteo T. Degiacomi
#
# Molearn is free software ;
# you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation ;
# either version 2 of the License, or (at your option) any later version.
# molearn is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY ;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with molearn ;
# if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from .protein_handler import get_convolutions

class Auto_potential():
    def __init__(self, frame, pdb_atom_names,
                padded_residues=False,
                method =('indexed', 'convolutional', 'roll')[2],
                device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), fix_h=False,alt_vdw=[], NB='repulsive',version=4):

        '''
        At instantiation will load amber parameters and create the necessary convolutions/indexs/rolls to calculate the energy of the molecule. Energy can be assessed with the
        ``Auto_conv_potential.get_loss(x)`` method

        :param frame: Example coordinates of the structure in a torch array. The interatomic distance will be used to determine the connectivity of the atoms.
            Coordinates should be of ``shape [3, N]`` where N is the number of atoms.
            If ``padded_residues = True`` then Coordinates should be of ``shape [R, M, 3]`` where R is the number of residues
            and M is the maximum number of atoms in a residue.
        :param pdb_atom_names: Array of ``shape [N, 2]`` containing the pdb atom names in ``pdb_atom_names[:, 0]`` and residue names in ``pdb_atom_names[:, 1]``. If
            ``padded_residues = True`` then should be ``shape [R, M, 2]``.
        :param padded_residues: If true the dataset should be formatted as ``shape [R, M, 3]`` where R is the number of residues and M is the maximum number of atoms.
            Padding should be ``nan``.
            **Note** only ``method = "indexed"`` is currently implemented currently for this.
        :param method: ``method = "convolutional"`` (currently experimental) method uses convolutions to calculate force (padded_residues=false only).

            ``method = "roll"`` method uses rolling and slicing to calculate force (padded_residues = false only)

            ``method = "indexed"`` (experimental) method is only impremented for padded_residues=True. Uses indexes to calculate forces.
        :param device: ``torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")``
        '''
        self.device = device
        self.method = method
        if padded_residues == True:
            if method == 'indexed':
                self._padded_indexed_init(frame, pdb_atom_names,version=version)
        else:
            if method == 'convolutional':
                self._convolutional_init(frame, pdb_atom_names, fix_h=fix_h, alt_vdw=alt_vdw, version=version)
            elif method == 'roll':
                self._roll_init(frame, pdb_atom_names, NB=NB, fix_h=fix_h,alt_vdw=alt_vdw, version=version)

    def get_loss(self, x):
        '''
            :param x: tensor of shape [B, 3, N] where B is batch_size, N is number of atoms. If padded_residues = True then tensor of shape [B, R, M, 3] where B is R is number of Residues and M is
                the maximum number of atoms in a residue.
            :returns: ``float`` Bond energy (average over batch)
            :returns: ``float`` Angle energy (average over batch)
            :returns: ``float`` Torsion energy (average over batch)
            :returns: ``float`` non-bonded energy (average over batch)
        '''
        if self.method == 'roll':
            return self._roll_loss(x)
        if self.method == 'convolutional':
            return self._convolutional_loss(x)
        if self.method == 'indexed':
            return self._padded_residues_loss(x)

    def _roll_init(self, frame, pdb_atom_names, NB='full', fix_h=False,alt_vdw=[], version=4):
        (b_masks, b_equil, b_force, b_weights,
            a_masks, a_equil, a_force, a_weights,
            t_masks, t_para, t_weights,
            vdw_R, vdw_e, vdw_14R, vdw_14e,
            q1q2, q1q2_14 )=get_convolutions(frame, pdb_atom_names, fix_slice_method=True, fix_h=fix_h,alt_vdw=alt_vdw,v=version)

        self.brdiff=[]
        self.br_equil=[]
        self.br_force=[]
        for i,j in enumerate(b_weights):
            atom1=j.index(1)
            atom2=j.index(-1)
            d = j.index(-1)-j.index(1)
            padding=len(j)-2
            self.brdiff.append(d)
            #b_equil[:,0] is just padding so can roll(-1,1) to get correct padding
            self.br_equil.append(torch.tensor(b_equil[i,padding-1:]).roll(-1).to(self.device).float())
            self.br_force.append(torch.tensor(b_force[i,padding-1:]).roll(-1).to(self.device).float())
        self.ardiff=[]
        self.arsign=[]
        self.arroll=[]
        self.ar_equil=[]
        self.ar_force=[]
        self.ar_masks=[]
        for i, j in enumerate(a_weights):
            atom1=j[0].index(1)
            atom2=j[0].index(-1)
            atom3=j[1].index(1)
            diff1=atom2-atom1
            diff2=atom2-atom3
            padding=len(j[0])-3
            self.arroll.append([min(atom1,atom2), min(atom2,atom3)])
            self.ardiff.append([abs(diff1)-1, abs(diff2)-1])
            self.arsign.append([diff1/abs(diff1), diff2/abs(diff2)])
            self.ar_equil.append(torch.tensor(a_equil[i,padding-2:]).roll(-2).to(self.device).float())
            self.ar_force.append(torch.tensor(a_force[i,padding-2:]).roll(-2).to(self.device).float())
        self.trdiff=[]
        self.trsign=[]
        self.trroll=[]
        self.tr_para=[]
        for i, j in enumerate(t_weights):
            atom1=j[0].index(1) #i-j 0
            atom2=j[0].index(-1) #i-j 2
            atom3=j[1].index(-1) #j-k 3
            atom4=j[2].index(1)  #l-k 4
            diff1=atom2-atom1 #ij 2
            diff2=atom3-atom2 #jk 1
            diff3=(atom4-atom3)*-1 #lk 1
            padding=len(j[0])-4
            self.trroll.append([min(atom1,atom2),min(atom2,atom3),min(atom3,atom4)])
            self.trsign.append([diff1/abs(diff1), diff2/abs(diff2), diff3/abs(diff3)])
            self.trdiff.append([abs(diff1)-1, abs(diff2)-1, abs(diff3)-1])
            self.tr_para.append(torch.tensor(t_para[i,padding-3:]).roll(-3,0).to(self.device).float())

        self.vdw_A = (vdw_e*(vdw_R**12)).to(self.device)
        self.vdw_B = (2*vdw_e*(vdw_R**6)).to(self.device)
        self.q1q2 = q1q2.to(self.device)

        self.get_loss = self._roll_loss
        if NB == 'full':
            self._nb_loss = self._cdist_nb_full
        elif NB == 'repulsive':
            self._nb_loss = self._cdist_nb

    def _convolutional_init(self, frame, pdb_atom_names, NB='full', fix_h=False,alt_vdw=[], version=4):
        (b_masks, b_equil, b_force, b_weights,
         a_masks, a_equil, a_force, a_weights,
         t_masks, t_para, t_weights,
         vdw_R, vdw_e, vdw_14R, vdw_14e,
         q1q2, q1q2_14 )=get_convolutions(frame, pdb_atom_names, fix_slice_method=False, fix_h=fix_h, alt_vdw=alt_vdw, v=version)

        self.b_equil  =torch.tensor(b_equil  ).to(self.device)
        self.b_force  =torch.tensor(b_force  ).to(self.device)
        self.b_weights=torch.tensor(b_weights).to(self.device)

        self.a_equil  =torch.tensor(a_equil  ).to(self.device).float()
        self.a_force  =torch.tensor(a_force  ).to(self.device).float()
        self.a_weights=torch.tensor(a_weights).to(self.device)
        self.a_masks  =torch.tensor(a_masks  ).to(self.device)

        self.t_para   =torch.tensor(t_para   ).to(self.device)
        self.t_weights=torch.tensor(t_weights).to(self.device)

        self.vdw_A = (vdw_e*(vdw_R**12)).to(self.device)
        self.vdw_B = (2*vdw_e*(vdw_R**6)).to(self.device)
        self.q1q2 = q1q2.to(self.device)

        self.get_loss = self._convolutional_loss
        if NB == 'full':
            self._nb_loss = self._cdist_nb_full
        elif NB == 'repulsive':
            self._nb_loss = self._cdist_nb

    def _padded_indexed_init(self, frame, pdb_atom_names, NB = 'full', version=2):
        from molearn import get_conv_pad_res
        (bond_idxs, bond_para,
             angle_idxs, angle_para, angle_mask, ij_jk,
             torsion_idxs, torsion_para, torsion_mask, ij_jk_kl,
             vdw_R, vdw_e,
             q1q2,) = get_conv_pad_res(frame, pdb_atom_names)#,v=version) doesn't take version

        self.bond_idxs = bond_idxs.to(self.device)
        self.bond_para = bond_para.to(self.device)
        self.angle_mask = angle_mask.to(self.device)
        self.ij_jk = ij_jk.to(self.device)
        self.angle_para = angle_para.to(self.device)
        self.torsion_mask = torsion_mask.to(self.device)
        self.ij_jk_kl = ij_jk_kl.to(self.device)
        self.torsion_para = torsion_para.to(self.device)
        self.vdw_A = (vdw_e*(vdw_R**12)).to(self.device)
        self.vdw_B = (2*vdw_e*(vdw_R**6)).to(self.device)
        self.q1q2 = q1q2.to(self.device)
        self.get_loss = self._padded_residues_loss
        self.relevant = self.bond_idxs.unique().to(self.device)
        if NB=='full':
            self._nb_loss = self._cdist_nb_full
        elif NB=='repulsive':
            self._nb_loss = self._cdist_nb

    def _convolutional_loss(self, x):
        bs = x.shape[0]
        bloss = self._conv_bond_loss(x)
        aloss = self._conv_angle_loss(x)
        tloss = self._conv_torsion_loss(x)
        nbloss = self._nb_loss(x)
        return bloss/bs, aloss/bs, tloss/bs, nbloss/bs

    def _roll_loss(self, x):
        bs = x.shape[0]
        bloss, aloss, tloss = self._roll_bond_angle_torsion_loss(x)
        nbloss = self._nb_loss(x)
        return bloss/bs, aloss/bs, tloss/bs, nbloss/bs

    def _padded_residues_loss(self, x):
        #x.shape [B, R, M, 3]
        x = x.view(x.shape[0], -1, 3)[:,]
        v = x[:,self.bond_idxs[:,1]]-x[:,self.bond_idxs[:,0]] #j-i == i->j
        bloss = (((v.norm(dim=2)-self.bond_para[:,0])**2)*self.bond_para[:,1]).sum()
        v1 = v[:,self.ij_jk[0]]*self.angle_mask[0].view(1,-1,1)
        v2 = v[:,self.ij_jk[1]]*self.angle_mask[1].view(1,-1,1)
        xyz=torch.sum(v1*v2, dim=2) / (torch.norm(v1, dim=2) * torch.norm(v2, dim=2))
        theta = torch.acos(torch.clamp(xyz, min=-0.999999, max=0.999999))
        aloss = (((theta-self.angle_para[:,0])**2)*self.angle_para[:,1]).sum()

        u1 = v[:,self.ij_jk_kl[0]]*self.torsion_mask[0].view(1, -1, 1)
        u2 = v[:,self.ij_jk_kl[1]]*self.torsion_mask[1].view(1, -1, 1)
        u3 = v[:,self.ij_jk_kl[2]]*self.torsion_mask[2].view(1, -1, 1)
        u12=torch.cross(u1,u2)
        u23=torch.cross(u2,u3)
        t3=torch.atan2(u2.norm(dim=2)*((u1*u23).sum(dim=2)),(u12*u23).sum(dim=2))
        p = self.torsion_para
        tloss=((p[:,1]/p[:,0])*(1+torch.cos((p[:,3]*t3.unsqueeze(2))-p[:,2]))).sum()
        bs = x.shape[0]
        return bloss/bs, aloss/bs, tloss/bs, self._nb_loss(x.permute(0,2,1))/bs

    def _cdist_nb_full(self, x, cutoff=9.0, mask=False):
        dmat = torch.cdist(x.permute(0,2,1),x.permute(0,2,1))
        dmat6 = (self._warp_domain(dmat, 1.9)**6)
        LJpB = self.vdw_B/dmat6
        LJpA = self.vdw_A/(dmat6**2)
        Cp = (self.q1q2/self._warp_domain(dmat, 0.4))
        return torch.nansum(LJpA-LJpB+Cp)

    def _cdist_nb(self, x, cutoff=9.0, mask=False):
        dmat = torch.cdist(x.permute(0,2,1),x.permute(0,2,1))
        LJp = self.vdw_A/(self._warp_domain(dmat, 1.9)**12)
        Cp = (self.q1q2/self._warp_domain(dmat, 0.4))
        return torch.nansum(LJp+Cp)

    def _warp_domain(self,x,k):
        return torch.nn.functional.elu(x-k, 1.0)+k

    def _conv_bond_loss(self, x):
        #x shape[B, 3, N]
        loss=torch.tensor(0.0).float().to(self.device)
        for i, weight in enumerate(self.b_weights):
            y = torch.nn.functional.conv1d(x, weight.view(1,1,-1).repeat(3,1,1).to(self.device), groups=3, padding=(len(weight)-2))
            loss+=(self.b_force[i]*((y.norm(dim=1)-self.b_equil[i])**2)).sum()
        return loss

    def _conv_angle_loss(self, x):
        #x shape[X, 3, N]
        loss=torch.tensor(0.0).float().to(self.device)
        for i, weight in enumerate(self.a_weights):
            v1 = torch.nn.functional.conv1d(x, weight[0].view(1,1,-1).repeat(3,1,1).to(self.device), groups=3, padding=(len(weight[0])-3))
            v2 = torch.nn.functional.conv1d(x, weight[1].view(1,1,-1).repeat(3,1,1).to(self.device), groups=3, padding=(len(weight[1])-3))
            xyz=torch.sum(v1*v2, dim=1) / (torch.norm(v1, dim=1) * torch.norm(v2, dim=1))
            theta = torch.acos(torch.clamp(xyz, min=-0.999999, max=0.999999))
            energy = (self.a_force[i]*((theta-self.a_equil[i])**2)).sum(dim=0)[self.a_masks[i]].sum()
            loss+=energy
        return loss

    def _conv_torsion_loss(self, x):
        #x shape[X, 3, N]
        loss=torch.tensor(0.0).float().to(self.device)
        for i, weight in enumerate(self.t_weights):
            b1 = torch.nn.functional.conv1d(x, weight[0].view(1,1,-1).repeat(3,1,1).to(self.device), groups=3, padding=(len(weight[0])-4))#i-j
            b2 = torch.nn.functional.conv1d(x, weight[1].view(1,1,-1).repeat(3,1,1).to(self.device), groups=3, padding=(len(weight[1])-4))#j-k
            b3 = torch.nn.functional.conv1d(x, weight[2].view(1,1,-1).repeat(3,1,1).to(self.device), groups=3, padding=(len(weight[2])-4))#l-k
            c32=torch.cross(b3,b2)
            c12=torch.cross(b1,b2)
            torsion=torch.atan2((b2*torch.cross(c32,c12)).sum(dim=1),
                                  b2.norm(dim=1)*((c12*c32).sum(dim=1)))
            p = self.t_para[i,:,:,:].unsqueeze(0)
            loss+=((p[:,:,1]/p[:,:,0])*(1+torch.cos((p[:,:,3]*torsion.unsqueeze(2))-p[:,:,2]))).sum()
        return loss

    def _roll_bond_angle_torsion_loss(self, x):
        #x.shape [5,3,2145]
        bloss = torch.tensor(0.0).float().to(self.device)
        aloss = torch.tensor(0.0).float().to(self.device)
        tloss = torch.tensor(0.0).float().to(self.device)
        v = []
        for i, diff in enumerate(self.brdiff):
            v.append(x-x.roll(-diff,2))
            bloss+=(((v[-1].norm(dim=1)-self.br_equil[i])**2)*self.br_force[i]).sum()

        for i, diff in enumerate(self.ardiff):
            v1 = self.arsign[i][0]*(v[diff[0]].roll(-self.arroll[i][0],2))
            v2 = self.arsign[i][1]*(v[diff[1]].roll(-self.arroll[i][1],2))
            xyz=torch.sum(v1*v2, dim=1) / (torch.norm(v1, dim=1) * torch.norm(v2, dim=1))
            theta = torch.acos(torch.clamp(xyz, min=-0.999999, max=0.999999)) 
            energy=(self.ar_force[i]*((theta-self.ar_equil[i])**2))
            sum_e = energy.sum()
            aloss+=(sum_e)

        for i, diff in enumerate(self.trdiff):
            b1 = self.trsign[i][0]*(v[diff[0]].roll(-self.trroll[i][0],2))
            b2 = self.trsign[i][1]*(v[diff[1]].roll(-self.trroll[i][1],2))
            b3 = self.trsign[i][2]*(v[diff[2]].roll(-self.trroll[i][2],2))
            c32=torch.cross(b3,b2)
            c12=torch.cross(b1,b2)
            torsion=torch.atan2((b2*torch.cross(c32,c12)).sum(dim=1),
                                  b2.norm(dim=1)*((c12*c32).sum(dim=1)))
            p = self.tr_para[i].unsqueeze(0)
            tloss+=( ((p[:,:,1]/p[:,:,0])*(1+torch.cos((p[:,:,3]*torsion.unsqueeze(2))-p[:,:,2]))).sum())
        return bloss,aloss,tloss

