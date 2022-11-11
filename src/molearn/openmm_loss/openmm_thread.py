import os
from openmm.unit import kelvin, picosecond
from openmm import Platform
from openmm.app import ForceField, PDBFile, Simulation, OBC2
from openmm.app import element as elem
import openmm
try:
    from torchexposedintegratorplugin import TorchExposedIntegrator
    from torchintegratorplugin import MyIntegrator
except ImportError:
    print('no plugin, wont be able to use openmm_loss')
import torch
from math import ceil
import numpy as np


from openmm.app.forcefield import _createResidueSignature
from openmm.app.internal import compiled

class ModifiedForceField(ForceField):
    def __init__(self, *args, alternative_residue_names = None, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(alternative_residue_names, dict):
            self._alternative_residue_names = alternative_residue_names
        else:
            self._alternative_residue_names = {'HIS':'HIE'}

    def _getResidueTemplateMatches(self, res, bondedToAtom, templateSignatures=None, ignoreExternalBonds=False, ignoreExtraParticles=False):
        """Return the templates that match a residue, or None if none are found.

        Parameters
        ----------
        res : Topology.Residue
            The residue for which template matches are to be retrieved.
        bondedToAtom : list of set of int
            bondedToAtom[i] is the set of atoms bonded to atom index i

        Returns
        -------
        template : _TemplateData
            The matching forcefield residue template, or None if no matches are found.
        matches : list
            a list specifying which atom of the template each atom of the residue
            corresponds to, or None if it does not match the template

        """
        template = None
        matches = None
        for matcher in self._templateMatchers:
            template = matcher(self, res, bondedToAtom, ignoreExternalBonds, ignoreExtraParticles)
            if template is not None:
                match = compiled.matchResidueToTemplate(res, template, bondedToAtom, ignoreExternalBonds, ignoreExtraParticles)
                if match is None:
                    raise ValueError('A custom template matcher returned a template for residue %d (%s), but it does not match the residue.' % (res.index, res.name))
                return [template, match]
        if templateSignatures is None:
            templateSignatures = self._templateSignatures
        signature = _createResidueSignature([atom.element for atom in res.atoms()])
        if signature in templateSignatures:
            allMatches = []
            for t in templateSignatures[signature]:
                match = compiled.matchResidueToTemplate(res, t, bondedToAtom, ignoreExternalBonds, ignoreExtraParticles)
                if match is not None:
                    allMatches.append((t, match))
            if len(allMatches) == 1:
                template = allMatches[0][0]
                matches = allMatches[0][1]
            elif len(allMatches) > 1:
                for i, (t, m) in enumerate(allMatches):
                    name = self._alternative_residue_names.get(res.name, res.name)
                    if name==t.name.split('-')[0] or 'N'+name==t.name.split('-')[0]:
                        template = t
                        matches = m
                        return [template, matches]
                # We found multiple matches.  This is OK if and only if they assign identical types and parameters to all atoms.
                t1, m1 = allMatches[0]

                for t2, m2 in allMatches[1:]:
                    if not t1.areParametersIdentical(t2, m1, m2):
                        raise Exception('Multiple non-identical matching templates found for residue %d (%s): %s.' % (res.index+1, res.name, ', '.join(match[0].name for match in allMatches)))
                template = allMatches[0][0]
                matches = allMatches[0][1]
        return [template, matches]




class OpenmmPluginScore():
    '''
    This will use the new Openmm Plugin to calculate forces and energy. The intention is that this will be fast enough to be able to calculate forces and energy during training.
    NB. The current torchintegratorplugin only supports float on GPU and double on CPU.
    '''
    
    def __init__(self, mol=None, xml_file = ['amber14-all.xml', 'implicit/obc2.xml'], platform = 'CUDA', remove_NB=False, alternative_residue_names = dict(HIS='HIE'), atoms=['CA', 'C', 'N', 'CB', 'O']):
        '''
        :param mol: (biobox.Molecule, default None) If pldataloader is not given, then a biobox object will be taken from this parameter. If neither are given then an error  will be thrown.
        :param data_dir: (string, default None) if pldataloader is not given then this will be used to find files such as 'variants.npy'
        :param xml_file: (string, default: "amber14-all.xml") xml parameter file
        :param platform: (string, default 'CUDA') either 'CUDA' or 'Reference'.
        :param remove_NB: (bool, default False) remove NonbondedForce, CustomGBForce, CMMotionRemover
        '''
        self.mol = mol
        if isinstance(xml_file,str):
            self.forcefield = ModifiedForceField(xml_file, alternative_residue_names = alternative_residue_names)
        elif len(xml_file)>0:
            self.forcefield = ModifiedForceField(*xml_file, alternative_residue_names = alternative_residue_names)
        else:
            raise ValueError(f'xml_file: {xml_file} needs to be a str or a list of str')
        tmp_file = 'tmp.pdb'
        self.atoms = atoms

        if atoms == 'no_hydrogen':
            self.ignore_hydrogen()
        else:
            self.atomselect(atoms)
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
        #self.integrator = MyIntegrator(300*kelvin, 1/picosecond, 0.002*picosecond)
        #self.integrator = TorchExposedIntegrator()
        #self.integrator = openmm.LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picosecond)
        from openmmtorch import VerletIntegrator
        self.integrator = VerletIntegrator(0.002*picosecond)

        self.platform = Platform.getPlatformByName(platform)
        self.simulation = Simulation(self.pdb.topology, self.system, self.integrator, self.platform)
        if platform == 'CUDA':
            self.platform.setPropertyValue(self.simulation.context, 'Precision', 'single')
        self.n_particles = self.simulation.context.getSystem().getNumParticles()
        self.simulation.context.setPositions(self.pdb.positions)
        self.get_score = self.get_energy
        print(self.simulation.context.getState(getEnergy=True).getPotentialEnergy()._value)
    
    def ignore_hydrogen(self):
        #ignore = ['ASH', 'LYN', 'GLH', 'HID', 'HIP', 'CYM', ]
        ignore = []
        for name, template in self.forcefield._templates.items():
            if name in ignore:
                continue
            patchData = ForceField._PatchData(name+'_remove_h', 1)
            
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
            self.forcefield.registerTemplatePatch(name, name+'_remove_h', 0)
            self.forcefield.registerPatch(patchData)

    def atomselect(self, atoms):
        for name, template in self.forcefield._templates.items():
            patchData = ForceField._PatchData(name+'_leave_only_'+'_'.join(atoms), 1)

            for atom in template.atoms:
                if atom.name not in atoms:
                    if atom.name not in patchData.allAtomNames:
                        patchData.allAtomNames.add(atom.name)
                        atomDescription = ForceField._PatchAtomData(atom.name)
                        patchData.deletedAtoms.append(atomDescription)
                    else:
                        raise ValueError()

            for bond in template.bonds:
                atom1 = template.atoms[bond[0]]
                atom2 = template.atoms[bond[1]]
                if atom1.name not in atoms or atom2.name not in atoms:
                    a1 = ForceField._PatchAtomData(atom1.name)
                    a2 = ForceField._PatchAtomData(atom2.name)
                    patchData.deletedBonds.append((a1, a2))
            self.forcefield.registerTemplatePatch(name, name+'_leave_only_'+'_'.join(atoms), 0)
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
        :param x: torch tensor, dtype = float, shape = [B, N, 3], device = any
        :returns: energy tensor, dtype = float, shape = [B], device  = any
        '''
        if x.device == torch.device('cpu'):
            force = np.zeros(x.shape)
            energy = np.zeros(x.shape[0])
            for i,t in enumerate(x):
                plugin.simulation.context.setPositions(t.numpy())
                state = plugin.simulation.context.getState(getForces=True, getEnergy=True)
                force[i] = state.getForces(asNumpy=True)
                energy[i] = state.getPotentialEnergy()._value
            force = torch.tensor(force).float()
            energy = torch.tensor(energy).float()
        else:
            #torch.cuda.synchronize(x.device)
            force, energy = plugin.execute(x)
            #torch.cuda.synchronize(x.device)
        ctx.save_for_backward(force)
        energy = energy.float().to(x.device)
        return energy

    @staticmethod
    def backward(ctx, grad_output):
        force = ctx.saved_tensors[0] # force shape [B, N, 3]
        #embed(header='23 openmm_loss_function')
        return None, -force*grad_output.view(-1,1,1)

class openmm_clamped_energy_function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, plugin, x, clamp):
        '''
        :param plugin: # OpenmmPluginScore instance
        :param x: torch tensor, dtype = float, shape = [B, N, 3], device = Cuda
        :returns: energy tensor, dtype = double, shape = [B], device CPU
        '''
        if x.device == torch.device('cpu'):
            force = np.zeros(x.shape)
            energy = np.zeros(x.shape[0])
            for i, t in enumerate(x):
                plugin.simulation.context.setPositions(t.numpy())
                state = plugin.simulation.context.getState(getForces=True, getEnergy=True)
                force[i] = state.getForces(asNumpy=True)
                energy[i] = state.getPotentialEnergy()._value
            force = torch.tensor(force).float()
            energy = torch.tensor(energy).float()
        else:
            force, energy = plugin.execute(x)

        force = torch.clamp(force, **clamp)
        ctx.save_for_backward(force)
        energy = energy.float().to(x.device)
        return energy

    @staticmethod
    def backward(ctx, grad_output):
        force = ctx.saved_tensors[0]
        return None, -force*grad_output.view(-1,1,1), None

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

class openmm_energy_process(torch.nn.Module):
    def __init__(self, mol, std, clamp = None, **kwargs):
        super().__init__()
        import multiprocessing as mp
        mp.set_start_method('spawn')
        from multiprocessing import Pool, get_context
        ctx = get_context('spawn')
        self.pool = Pool(initializer=set_global_score, initargs=(OpenmmPluginScore, dict(mol=mol, **kwargs)), processes=1)
        self.std = std/10
        self.clamp = clamp
        if self.clamp is not None:
            fail_here
            self.forward = self._clamp_forward
        else:
            self.forward = self._forward

    def _forward(self, x):
        '''
        :param x: torch tensor dtype=torch.float, device=CUDA, shape B, 3, N 
        :returns: torch energy tensor dtype should be float and on same device as x
        '''
        _x = (x*self.std).permute(0,2,1).contiguous()
        energy = alt_openmm_energy_function.apply(self.pool, _x)
        return energy

    def _clamp_forward(self, x):
        '''
        :param x: torch tensor dtype=torch.float, device=CUDA, shape B, 3, N 
        :returns: torch energy tensor dtype should be float and on same device as x
        '''
        _x = (x*self.std).permute(0,2,1).contiguous()
        energy = openmm_clamped_energy_function.apply(self.openmmplugin, _x, self.clamp)
        return energy

def set_global_score(score, kwargs):
    global worker_score
    worker_score = score(**kwargs)

def process_score(args):
    worker_score.get_energy(*args)
    return

class alt_openmm_energy_function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, pool, x):
        '''
        :param plugin: # OpenmmPluginScore instance
        :param x: torch tensor, dtype = float, shape = [B, N, 3], device = any
        :returns: energy tensor, dtype = float, shape = [B], device  = any
        '''
        if x.device == torch.device('cpu'):
            fail_here
            force = np.zeros(x.shape)
            energy = np.zeros(x.shape[0])
            for i,t in enumerate(x):
                plugin.simulation.context.setPositions(t.numpy())
                state = plugin.simulation.context.getState(getForces=True, getEnergy=True)
                force[i] = state.getForces(asNumpy=True)
                energy[i] = state.getPotentialEnergy()._value
            force = torch.tensor(force).float()
            energy = torch.tensor(energy).float()
        else:
            force = torch.zeros_like(x)
            energy = torch.zeros(x.shape[0], device = torch.device('cpu'), dtype=torch.double)
            result = pool.apply_async(process_score, ((x.data_ptr(), force.data_ptr(), energy.data_ptr(), x.shape[1], x.shape[0]),))
            result.get()
        embed(header='test')
        ctx.save_for_backward(force)
        energy = energy.float().to(x.device)
        return energy

    @staticmethod
    def backward(ctx, grad_output):
        force = ctx.saved_tensors[0] # force shape [B, N, 3]
        #embed(header='23 openmm_loss_function')
        return None, -force*grad_output.view(-1,1,1)


