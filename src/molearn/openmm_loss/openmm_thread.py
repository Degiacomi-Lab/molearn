import os
from openmm.unit import kelvin, picosecond
from openmm import Platform
from openmm.app import ForceField, PDBFile, Simulation, OBC2
from openmm.app import element as elem
import openmm
from torchintegratorplugin import MyIntegrator

import torch
from math import ceil
from IPython import embed

class OpenmmPluginScore():
    '''
    This will use the new Openmm Plugin to calculate forces and energy. The intention is that this will be fast enough to be able to calculate forces and energy during training.
    NB. The current torchintegratorplugin only supports float on GPU and double on CPU.
    '''
    
    def __init__(self, mol=None, xml_file = ['amber14-all.xml', 'implicit/obc2.xml'], platform = 'CUDA', remove_NB=False):
        '''
        :param mol: (biobox.Molecule, default None) If pldataloader is not given, then a biobox object will be taken from this parameter. If neither are given then an error  will be thrown.
        :param data_dir: (string, default None) if pldataloader is not given then this will be used to find files such as 'variants.npy'
        :param xml_file: (string, default: "amber14-all.xml") xml parameter file
        :param platform: (string, default 'CUDA') either 'CUDA' or 'Reference'.
        :param remove_NB: (bool, default False) remove NonbondedForce, CustomGBForce, CMMotionRemover
        '''
        self.mol = mol
        if isinstance(xml_file,str):
            self.forcefield = ForceField(xml_file)
        elif len(xml_file)>0:
            self.forcefield = ForceField(*xml_file)
        else:
            raise ValueError(f'xml_file: {xml_file} needs to be a str or a list of str')
        tmp_file = 'tmp.pdb'
        
        self.ignore_hydrogen()

        #save pdb and reload in modeller
        self.mol.write_pdb(tmp_file)
        self.pdb = PDBFile(tmp_file)
        templates, unique_unmatched_residues = self.forcefield.generateTemplatesForUnmatchedResidues(self.pdb.topology)
        self.system = self.forcefield.createSystem(self.pdb.topology)
        if remove_NB:
            forces = self.system.getForces()
            for idx in reversed(range(len(forces))):
                force = forces[idx]
                if isinstance(force, (#openmm.PeriodicTorsionForce,
                                      openmm.CustomGBForce,
                                      openmm.NonbondedForce,
                                      openmm.CMMotionRemover)):
                    self.system.removeForce(idx)
        self.integrator = MyIntegrator(300*kelvin, 1/picosecond, 0.002*picosecond)

        self.platform = Platform.getPlatformByName(platform)
        self.simulation = Simulation(self.pdb.topology, self.system, self.integrator, self.platform)
        if platform == 'CUDA':
            self.platform.setPropertyValue(self.simulation.context, 'Precision', 'single')
        self.n_particles = self.simulation.context.getSystem().getNumParticles()
        self.simulation.context.setPositions(self.pdb.positions)
        self.get_score = self.get_energy
        print(self.simulation.context.getState(getEnergy=True).getPotentialEnergy()._value)
    
    def ignore_hydrogen(self):
        ignore = ['ASH', 'LYN', 'GLH', 'HID', 'HIP', 'CYM', ]
        for name, template in self.forcefield._templates.items():
            if name in ignore:
                continue
            patchData = ForceField._PatchData(name+'_noh', 1)

            for atom in template.atoms:
                if atom.element is elem.hydrogen:
                    if atom.name not in patchData.allAtomNames:
                        patchData.allAtomNames.add(atom.name)
                        atomDescription = ForceField._PatchAtomData(atom.name)
                        patchData.deletedAtoms.append(atomDescription)
                    else:
                        raise ValueError()
            for bond in template.bonds:
                atom1 = template.atoms[bond[0]]
                atom2 = template.atoms[bond[1]]
                if atom1.element is elem.hydrogen or atom2.element is elem.hydrogen:
                    a1 = ForceField._PatchAtomData(atom1.name)
                    a2 = ForceField._PatchAtomData(atom2.name)
                    patchData.deletedBonds.append((a1, a2))
            self.forcefield.registerTemplatePatch(name, name+'_noh', 0)
            self.forcefield.registerPatch(patchData)

    def get_energy(self, pos_ptr, force_ptr, energy_ptr, n_particles, batch_size):
        '''
        :param pos_ptr: tensor.data_ptr()
        :param force_ptr: tensor.data_ptr()
        :param energy_ptr: tensor.data_ptr()
        :param n_particles: int
        :param batch_size: int
        '''
        assert n_particles == self.n_particles
        self.integrator.torchMultiStructureE(pos_ptr, force_ptr, energy_ptr, n_particles, batch_size)
        return True

    def execute(self, x):
        '''
        :param x: torch tensor shape [b, N, 3]. dtype=float. device = gpu
        '''
        force = torch.zeros_like(x)
        energy = torch.zeros(x.shape[0], device = torch.device('cpu'), dtype=torch.double)
        self.get_energy(x.data_ptr(), force.data_ptr(), energy.data_ptr(), x.shape[1], x.shape[0])
        return force, energy

class openmm_energy_function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, plugin, x):
        '''
        :param plugin: # OpenmmPluginScore instance
        :param x: torch tensor, dtype = float, shape = [B, N, 3], device = Cuda
        :returns: energy tensor, dtype = double, shape = [B], device CPU
        '''
        force, energy = plugin.execute(x)
        ctx.save_for_backward(force)
        energy = energy.float().to(x.device)
        return energy

    @staticmethod
    def backward(ctx, grad_output):
        force = ctx.saved_tensors[0] # force shape [B, N, 3]
        #embed(header='23 openmm_loss_function')
        return None, -force*grad_output.view(-1,1,1)

class openmm_energy(torch.nn.Module):
    def __init__(self, mol, std, clamp = None, **kwargs):
        super().__init__()
        self.openmmplugin = OpenmmPluginScore(mol, **kwargs)
        self.std = std/10
        self.clamp = clamp
        if self.clamp is not None:
            self.forward = self._clamp_forward
        else:
            self.forward = self._forward

    def _forward(self, x):
        '''
        :param x: torch tensor dtype=torch.float, device=CUDA, shape B, 3, N 
        :returns: torch energy tensor dtype should be float and on same device as x
        '''
        _x = (x*self.std).permute(0,2,1).contiguous()
        energy = openmm_energy_function.apply(self.openmmplugin, _x)
        return energy

    def _clamp_forward(self, x):
        '''
        :param x: torch tensor dtype=torch.float, device=CUDA, shape B, 3, N 
        :returns: torch energy tensor dtype should be float and on same device as x
        '''
        _x = (x*self.std).permute(0,2,1).contiguous()
        energy = openmm_clamped_energy_function.apply(self.openmmplugin, _x, self.clamp)
        return energy
