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
import time
from copy import deepcopy
import biobox
import os
import pkg_resources


def read_lib_file(file_name, amber_atoms, atom_charge, connectivity):
    try:
        f_location = pkg_resources.resource_filename('molearn', 'parameters')
        f_in = open(f'{f_location}/{file_name}')
        print('File %s opened' % file_name)
    except Exception as ex:
        raise Exception('ERROR: file %s not found!' % file_name)


    lines = f_in.readlines()
    depth = 0
    indexs = {}
    for tline in lines:
        if tline.split()==['!!index', 'array', 'str']:
            depth+=1
            for line in lines[depth:]:
                if line[0]!=' ':
                    break
                contents = line.split()
                if len(contents)!=1 and len(contents[0])!=5:
                    break
                res=contents[0]
                if res[0]=='"' and res[-1] =='"':
                   amber_atoms[res[1:-1]]={}
                   atom_charge[res[1:-1]]={}
                   indexs[res[1:-1]]={}
                   connectivity[res[1:-1]]={}
                else:
                   raise Exception(('I was expecting something of the form'
                                       +'"XXX" but got %s instead' % res))
                depth+=1
            break
        depth+=1
    for i, tline in enumerate(lines):
        entry, res, unit_atoms, unit_connectivity = tline[0:7],tline[7:10], tline[10:22], tline[10:29]
        if entry=='!entry.' and unit_atoms=='.unit.atoms ':
            depth=i+1
            for line in lines[depth:]:
                if line[ 0]!=' ':
                    break
                contents = line.split()
                if len(contents)<3 and len(contents[0])>4 and len(contents[1])>4:
                    break
                pdb_name, amber_name,_,_,_,index,element_number,charge = contents
                if (pdb_name[0]=='"' and pdb_name[-1]=='"'
                    and amber_name[0]=='"' and amber_name[-1]=='"'):
                    amber_atoms[res][contents[0][1:-1]] = contents[1][1:-1]
                    atom_charge[res][amber_name[1:-1]] = float(charge)
                    #indexs[res][amber_name[1:-1]] = int(index)
                    indexs[res][int(index)] = pdb_name[1:-1]
                    connectivity[res][pdb_name[1:-1]] = []
                else:
                   raise Exception(('I was expecting something of the form'
                                       +'"XXX" but got %s instead' % res))
        elif entry=='!entry.' and unit_connectivity=='.unit.connectivity ':
            depth=i+1
            for line in lines[depth:]:
                if line[0]!=' ':
                    break
                contents = line.split()
                if len(contents)!=3:
                    break
                a1,a2,flag = contents
                connectivity[res][indexs[res][int(a1)]].append(indexs[res][int(a2)])
                connectivity[res][indexs[res][int(a2)]].append(indexs[res][int(a1)])

def get_amber_parameters(order=False, radians=True):

    file_names =('amino12.lib',
                 'parm10.dat',
                 'frcmod.ff14SB')

              #amber19 is dangerous because they've replaced parameters with cmap
    ###### pdb atom names to amber atom names using amino19.lib ######
    amber_atoms          = {} # knowledge[res][pdb_atom] = amber_atom
    atom_mass            = {}
    atom_polarizability  = {}
    bond_force           = {}
    bond_equil           = {}
    angle_force          = {}
    angle_equil          = {}
    torsion_factor       = {}
    torsion_barrier      = {}
    torsion_phase        = {}
    torsion_period       = {}
    improper_factor      = {}
    improper_barrier     = {}
    improper_phase       = {}
    improper_period      = {}

    other_parameters     = {}
    other_parameters['vdw_potential_well_depth']={}
    other_parameters['H_bond_10_12_parameters']={}
    other_parameters['equivalences']={}
    other_parameters['charge']={}
    other_parameters['connectivity']={}
    read_lib_file(file_names[0],amber_atoms,other_parameters['charge'], other_parameters['connectivity'])

    try:
        f_location = pkg_resources.resource_filename('molearn', 'parameters')
        f_in = open(f'{f_location}/{file_names[1]}')
        print('File %s opened' % file_names[1])
    except Exception as ex:
        raise Exception('ERROR: file %s not found!' % file_names[1])

    #section 1 title
    line = f_in.readline()
    print(line)

    amber_card_type_2(f_in, atom_mass, atom_polarizability)
    amber_card_type_3(f_in)
    amber_card_type_4(f_in, bond_force, bond_equil)
    amber_card_type_5(f_in, angle_force, angle_equil)
    amber_card_type_6(f_in, torsion_factor, torsion_barrier, torsion_phase, torsion_period)
    amber_card_type_7(f_in, improper_factor,
                      improper_barrier, improper_phase, improper_period)
    amber_card_type_8(f_in, other_parameters)
    amber_card_type_9(f_in, other_parameters)
    for line in f_in:
        if len(line.split())>1:
            if line.split()[1]=='RE':
                amber_card_type_10B(f_in, other_parameters)
        elif line[0:3]=='END':
            print('parameters loaded')
    f_in.close()

    #open frcmod file, should be identifcal format but missing any or all cards
    try:
        f_location = pkg_resources.resource_filename('molearn', 'parameters')
        f_in = open(f'{f_location}/{file_names[2]}')
        print('File %s opened' % file_names[2])
    except Exception as ex:
        raise Exception('ERROR: file %s not found!' % file_names[2])


    #section 1 title
    line = f_in.readline()
    print(line)

    for line in f_in:
        if line[:4]=='MASS':
            amber_card_type_2(f_in, atom_mass, atom_polarizability)
        if line[:4]=='BOND':
            amber_card_type_4(f_in, bond_force, bond_equil)
        if line[:4]=='ANGL':
            amber_card_type_5(f_in, angle_force, angle_equil)
        if line[:4]=='DIHE':
            amber_card_type_6(f_in, torsion_factor, torsion_barrier, torsion_phase, torsion_period)
        if line[:4]=='IMPR':
            amber_card_type_7(f_in, improper_factor,
                              improper_barrier, improper_phase, improper_period)
        if line[:4]=='HBON':
            amber_card_type_8(f_in, other_parameters)
        if line[:4]=='NONB':
            amber_card_type_10B(f_in, other_parameters)
        if line[:4]=='CMAP':
            print('Yeah, Im not bothering to implement cmap')
        elif line[0:3]=='END':
            print('parameters loaded')
    f_in.close()

    if radians:
        for angle in angle_equil:
            angle_equil[angle]=np.deg2rad(angle_equil[angle])
        for torsion in torsion_phase:
            torsion_phase[torsion]=list(np.deg2rad(torsion_phase[torsion]))

    return (amber_atoms, atom_mass, atom_polarizability, bond_force, bond_equil,
            angle_force, angle_equil, torsion_factor, torsion_barrier, torsion_phase,
            torsion_period, improper_factor, improper_barrier, improper_phase,
            improper_period, other_parameters)

def amber_card_type_2(f_in, atom_mass, atom_polarizability):

    #section 2 input for atom symbols and masses
    for line in f_in:
        if line=='\n' or line.strip()=='':
            break
        atom = line[0:2].strip()
        contents = line[2:24].split()
        if len(contents)==2:
            mass, polarizability = float(contents[0]),float(contents[1])
            atom_mass[atom]=mass
            atom_polarizability[atom]=polarizability
        elif len(contents)==1: # sometimes a polarizability is not listed
            mass = float(contents[0])
            atom_mass[atom]=mass
            atom_polarizability[atom]=polarizability
        else:
            raise Exception('Should be 2A, X, F10.2, F10.2, comments but got %s' % line)

def amber_card_type_3(f_in):
    #section 3  input for atom symbols that are hydrophilic
    line = f_in.readline()

def amber_card_type_4(f_in, bond_force, bond_equil, order=False):
    #section 4  bond length paramters
    for line in f_in:
        if line=='\n' or line.strip()=='':
            break
        atom1 = line[0:2]. strip()
        atom2 = line[3:5].strip()
        if order:
            bond = tuple(sorted((atom1, atom2))) # put in alphabetical order
        else:
            bond = (atom1, atom2)
        contents = line[5:25].split()
        if len(contents)!=2:
            raise Exception('Expected 2 floats but got %s' % line[6:26])
        force_constant, equil_length = float(contents[0]), float(contents[1])
        bond_force[bond]=force_constant
        bond_equil[bond]=equil_length
        #this should throw an error if there are not 

def amber_card_type_5(f_in, angle_force, angle_equil, order=False):
    #section 5
    for line in f_in:
        if line=='\n' or line.strip()=='':
            break
        atom1 = line[0:2].strip()
        atom2 = line[3:5].strip()
        atom3 = line[6:8].strip()
        if order:
            sorted13 = sorted((atom1, atom3))
            angle = (sorted13[0], atom2, sorted13[1])
            # I want it sorted alphabetically by 1-3 atoms
        else:
            angle=(atom1, atom2, atom3)
        contents = line[8:28].split()
        if len(contents)!=2:
            raise Exception('Expected 2 floats but got %s' % line[6:26])
        force_constant, equil_angle = float(contents[0]), float(contents[1])
        angle_force[angle]=force_constant
        angle_equil[angle]=equil_angle

def amber_card_type_6(f_in, torsion_factor, torsion_barrier, torsion_phase,
                      torsion_period, order=False):
    #secti on 6 torsion / proper dihedral
    for line in f_in:
        if line=='\n' or line.strip()=='':
            break
        atom1 = line[0:2].strip()
        atom2 = line[3:5].strip()
        atom3 = line[6:8].strip()
        atom4 = line[9:11].strip()
        if order:
            sort23 = sorted([(atom2, atom1), (atom3, atom4)], key=lambda x: x[0])
            torsion = tuple( (sort23[0][1], sort23[0][0], sort23[1][0], sort23[1][1]) )
        else:
            torsion = (atom1, atom2, atom3, atom4)
        contents = line[11:55].split()
        if len(contents)!=4:
            raise Exception('I wanted four values here?')
        #the actual torsion potential is (barrier/factor)*(1+cos(period*phi-phase))
        if torsion in torsion_period:
            if torsion_period[torsion][-1]>0:
                torsion_factor[torsion]  = [int(contents[0])  ]
                torsion_barrier[torsion] = [float(contents[1])]
                torsion_phase[torsion]   = [float(contents[2])]
                torsion_period[torsion]  = [float(contents[3])]
            elif torsion_period[torsion][-1]<0:
                torsion_factor[torsion].append(   int(contents[0]))
                torsion_barrier[torsion].append(float(contents[1]))
                torsion_phase[torsion].append(  float(contents[2]))
                torsion_period[torsion].append( float(contents[3]))
        else:
            torsion_factor[torsion]  = [int(contents[0])  ]
            torsion_barrier[torsion] = [float(contents[1])]
            torsion_phase[torsion]   = [float(contents[2])]
            torsion_period[torsion]  = [float(contents[3])]

def amber_card_type_7(f_in, improper_factor, improper_barrier,
                      improper_phase, improper_period, order=False):
    #section 7 improper dihedrals 
    for line in f_in:
        if line=='\n' or line.strip()=='':
            break
        atom1 = line[0:2].strip()
        atom2 = line[3:5].strip()
        atom3 = line[6:8].strip()
        atom4 = line[9:11].strip()
        if order:
            sort23 = sorted([(atom2, atom1), (atom3, atom4)], key=lambda x: x[0])
            torsion = tuple( (sort23[0][1], sort23[0][0], sort23[1][0], sort23[1][1]) )
        else:
            torsion = (atom1, atom2, atom3, atom4)
        contents = line[11:55].split()
        if len(contents)==3:
            improper_barrier[torsion] = float(contents[0])
            improper_phase[torsion]   = float(contents[1])
            improper_period[torsion]  = float(contents[2])
        elif len(contents)==4:
            raise Exception('This seems allowed in the doc but doesnt appear in reality')
            improper_factor[torsion]  = int(contents[0])
            improper_barrier[torsion] = float(contents[1])
            improper_phase[torsion]   = float(contents[2])
            improper_period[torsion]  = float(contents[3])
        #the actual torsion potential is (barrier/factor)*(1+cos(period*phi-phase))
        #it seems improper potential don't divide by the factor

def amber_card_type_8(f_in, other_parameters, order=False):
    #section 8  H-bond 10-12 potential parameters
    for line in f_in:
        if line=='\n' or line.strip()=='':
            break
        atom1 = line[2:4].strip()
        atom2 = line[6:8].strip()
        if order:
            pair = tuple(sorted((atom1, atom2)))
        else:
            pair = (atom1, atom2)
        contents = line[8:].split()
        other_parameters['H_bond_10_12_parameters'][pair]=contents

def amber_card_type_9(f_in, other_parameters):
    #section 9 equi valencing atom symbols for non-bonded 6-12 potential parameters
    for line in f_in:
        if line=='\n' or line.strip()=='':
            break
        contents = line.split()
        other_parameters['equivalences'][contents[0]]=contents

def amber_card_type_10B(f_in, other_parameters):
    #section 10 6-12  potential parameters
    for line in f_in:
        if line== '\n' or line.strip()=='':
            break
        contents = line.split()
        other_parameters['vdw_potential_well_depth'][contents[0]] = [float(i) for i in contents[1:3]]

def get_convolutions(dataset, pdb_atom_names,
                              atom_label=('set','string')[0],
                              perform_checks=True,
                              v=5,
                              order=False,
                              return_type=['mask','idxs'][1],
                              absolute_torsion_period=True,
                              NB=('matrix',)[0],
                              fix_terminal=True,
                              fix_charmm_residues=True,
                              fix_slice_method=False,
                              fix_h=False,
                              alt_vdw = [],
                              permitivity=1.0
                             ):
    '''
    ##INPUTS##

    dataset:         one frame of a trajectory of shape [3, N]

    pdb_atom_names:  should be an array of shape [N,2]
                     pdb_atom_names[:,0] is the pdb_atom_names and
                     pdb_atom_names[:,1] is the residue names

    atom_label:      (default, 'set') deprecated and broken for anything other than 'set'

    perform_checks:  No longer works so has been removed

    v:               (default, 5) atom_selection version, bonds are determined by interatomic
                     distance, with v=2. v=1 shouldn't be used except in specific cirumstances. v=5 is using connectivity from amber parameters.

    order:           (bool, default false) are atoms ordered, I think I've fixed this so that
                     it shouldn't matter either way but keep as False.

    return_type:     (option now removed)


    ##OUTPUTS##
    convolution output shape N* will be N-(conv length -1)+padding

    bond_masks, b_equil, b_force: shape [number of convolutions, N*]

    bond_weights: shape[number of convolutions, conv_size]

    angle_masks, a_equil, a_force:  shape [number of convolutions,  conv_size]

    angle_weights: shape[number of convolutions, 2, conv_size]

    torsion_masks: shape[number of convolution, 3, conv_size]

    t_para:        shape[num of convs, N*, 4, max number torsion parameters ]

    tornsion_weigths: shape [number of convolutions, 3, conv_size]

    '''


    #get amber parameters
    (amber_atoms, atom_mass, atom_polarizability, bond_force, bond_equil,
    angle_force, angle_equil, torsion_factor, torsion_barrier, torsion_phase,
    torsion_period, improper_factor, improper_barrier, improper_phase,
    improper_period, other_parameters) =  get_amber_parameters()
    if fix_terminal:
        pdb_atom_names[pdb_atom_names[:,0]=='OXT',0]='O'
    if fix_charmm_residues:
        pdb_atom_names[pdb_atom_names[:,1]=='HSD',1]='HID'
        pdb_atom_names[pdb_atom_names[:,1]=='HSE',1]='HIE'
        for i in np.unique(pdb_atom_names[:,2]):
            res_mask = pdb_atom_names[:,2]==i
            if (pdb_atom_names[res_mask, 1]=='HIS').all(): # if a HIS residue
                if (pdb_atom_names[res_mask, 0]=='HD1').any() and (pdb_atom_names[res_mask, 0]=='HE2').any():
                    pdb_atom_names[res_mask, 1]='HIP'
                elif (pdb_atom_names[res_mask, 0]=='HD1').any():
                    pdb_atom_names[res_mask, 1]='HID'
                elif (pdb_atom_names[res_mask, 0]=='HE2').any():
                    pdb_atom_names[res_mask, 1]='HIE'
        #if any HIS are remaining it doesn't matter which because the H is dealt with above
        pdb_atom_names[pdb_atom_names[:,1]=='HIS',1]='HIE'
    if fix_h:
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HB1', pdb_atom_names[:,1]=='MET'),0]='HB3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HG1', pdb_atom_names[:,1]=='MET'),0]='HG3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HB1', pdb_atom_names[:,1]=='ASN'),0]='HB3'
        pdb_atom_names[pdb_atom_names[:,0]=='HN',0]='H'
        pdb_atom_names[pdb_atom_names[:,0]=='1HD2',0]='HD21'
        pdb_atom_names[pdb_atom_names[:,0]=='2HD2',0]='HD22'
        pdb_atom_names[pdb_atom_names[:,0]=='1HG2',0]='HG21'
        pdb_atom_names[pdb_atom_names[:,0]=='2HG2',0]='HG22'
        pdb_atom_names[pdb_atom_names[:,0]=='3HG2',0]='HG23'
        pdb_atom_names[pdb_atom_names[:,0]=='3HG1',0]='HG13'
        pdb_atom_names[pdb_atom_names[:,0]=='1HG1',0]='HG11'
        pdb_atom_names[pdb_atom_names[:,0]=='2HG1',0]='HG12'
        pdb_atom_names[pdb_atom_names[:,0]=='1HD1',0]='HD11'
        pdb_atom_names[pdb_atom_names[:,0]=='2HD1',0]='HD12'
        pdb_atom_names[pdb_atom_names[:,0]=='3HD1',0]='HD13'
        pdb_atom_names[pdb_atom_names[:,0]=='3HD2',0]='HD23'
        pdb_atom_names[pdb_atom_names[:,0]=='1HH1',0]='HH11'
        pdb_atom_names[pdb_atom_names[:,0]=='2HH1',0]='HH12'
        pdb_atom_names[pdb_atom_names[:,0]=='1HH2',0]='HH21'
        pdb_atom_names[pdb_atom_names[:,0]=='2HH2',0]='HH22'
        pdb_atom_names[pdb_atom_names[:,0]=='1HE2',0]='HE21'
        pdb_atom_names[pdb_atom_names[:,0]=='2HE2',0]='HE22'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HG11', pdb_atom_names[:,1]=='ILE'),0]='HG13'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='CD', pdb_atom_names[:,1]=='ILE'),0]='CD1'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HD1', pdb_atom_names[:,1]=='ILE'),0]='HD11'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HD2', pdb_atom_names[:,1]=='ILE'),0]='HD12'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HD3', pdb_atom_names[:,1]=='ILE'),0]='HD13'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HB1', pdb_atom_names[:,1]=='PHE'),0]='HB3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HB1', pdb_atom_names[:,1]=='GLU'),0]='HB3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HG1', pdb_atom_names[:,1]=='GLU'),0]='HG3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HB1', pdb_atom_names[:,1]=='LEU'),0]='HB3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HB1', pdb_atom_names[:,1]=='ARG'),0]='HB3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HG1', pdb_atom_names[:,1]=='ARG'),0]='HG3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HD1', pdb_atom_names[:,1]=='ARG'),0]='HD3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HB1', pdb_atom_names[:,1]=='ASP'),0]='HB3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HA1', pdb_atom_names[:,1]=='GLY'),0]='HA3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HB1', pdb_atom_names[:,1]=='LYS'),0]='HB3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HG1', pdb_atom_names[:,1]=='LYS'),0]='HG3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HD1', pdb_atom_names[:,1]=='LYS'),0]='HD3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HE1', pdb_atom_names[:,1]=='LYS'),0]='HE3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HB1', pdb_atom_names[:,1]=='TYR'),0]='HB3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HB1', pdb_atom_names[:,1]=='HIP'),0]='HB3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HB1', pdb_atom_names[:,1]=='SER'),0]='HB3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HG1', pdb_atom_names[:,1]=='SER'),0]='HG'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HB1', pdb_atom_names[:,1]=='PRO'),0]='HB3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HG1', pdb_atom_names[:,1]=='PRO'),0]='HG3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HD1', pdb_atom_names[:,1]=='PRO'),0]='HD3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HB1', pdb_atom_names[:,1]=='LEU'),0]='HB3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HB1', pdb_atom_names[:,1]=='GLN'),0]='HB3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HG1', pdb_atom_names[:,1]=='GLN'),0]='HG3'
        pdb_atom_names[np.logical_and(pdb_atom_names[:,0]=='HB1', pdb_atom_names[:,1]=='TRP'),0]='HB3'
    #writes termini as H because we haven't loaded in termini parameters
    atom_names = [[amber_atoms[res][atom],res, resid] if atom not in ['H2', 'H3'] else [amber_atoms[res]['H'], res, resid] for atom, res, resid in pdb_atom_names ]
    p_atom_names = [[atom,res, resid] if atom not in ['H2', 'H3'] else ['H', res, resid] for atom, res, resid in pdb_atom_names ]
    
    #atom_names = [[amber_atoms[res][atom],res] for atom, res, resid in pdb_atom_names ]
    atom_charges=[other_parameters['charge'][res][atom] for atom, res, resid in atom_names]
    if NB == 'matrix':
        equiv_t=other_parameters['equivalences']
        vdw_para = other_parameters['vdw_potential_well_depth']
        #switch these around so that values point to key
        equiv = {}
        for i in equiv_t.keys():
            j = equiv_t[i]
            for k in j:
                equiv[k]=i
        atom_R = torch.tensor([vdw_para[equiv.get(atom,atom)][0] for atom, res, resid in atom_names]) #radius
        atom_e = torch.tensor([vdw_para[equiv.get(atom,atom)][1] for atom, res, resid in atom_names]) #welldepth

    print('Determining bonds')
    version = v # method of selecting bonded atoms
    N = dataset.shape[1] #2145
    
    cmat=(torch.nn.functional.pdist((dataset).permute(1,0))).cpu().numpy()
    if version == 1:
        bond_idxs=np.argpartition(cmat, (N-1,N))
        #this will work for any non cyclic monomeric protein
        #that in mind will break if enough proline atoms to make a cycle are selected
        bond_idxs, u = bond_idxs[:N-1], bond_idxs[N-1]
        if cmat[u]-cmat[bond_idxs[-1]]<0.25:
            raise Exception("WARNING: May not have correctly selected the bonded distances: value "
                            +(cmat[u]-cmat[bond_idxs[-1]])+
                            "should be roughly between 0.42 and 0.57 (>0.25)" )# should be 0.42-0.57
            version+=1 #try version 2 instead
        mid = cmat[bond_idxs[-1]]+((cmat[u]-cmat[bond_idxs[-1]])/2) #mid point
        full_mask = (cmat<mid).astype('int8')
    if version == 2:
        full_mask = (cmat<(1.643+2.129)/2).astype('int8')
        bond_idxs = np.where(full_mask)[0] # for some reason returns tuple with one array
    if version == 3:
        if alt_vdw:
            vdw = torch.tensor(alt_vdw)
            max_bond_dist = (0.6*(vdw.view(1,-1)+vdw.view(-1,1)))
            cdist = torch.cdist(dataset.T,dataset.T)
            i,j = np.where((max_bond_dist>cdist).triu(diagonal=1).numpy())
            remove = np.where(np.abs(j-i)>30)
            max_bond_dist[i[remove],j[remove]]=0.0
            max_bond_dist = max_bond_dist.numpy()
        else:
            max_bond_dist = (0.6*(atom_R.view(1,-1)+atom_R.view(-1,1))).cpu().numpy()
        max_bond_dist = max_bond_dist[np.where(np.triu(np.ones((N,N)),k=1))]
        full_mask = np.greater(max_bond_dist,cmat)
        bond_idxs = np.where(full_mask)[0] # for some reason returns tuple with one array
    if version == 4:
        #fix_hydrogens = [[atom, res, resid] for atom, res, resid in pdb_atom_names if atom in ['H2', 'H3']]
        connectivity = other_parameters['connectivity']
        bond_types = []
        bond_idxs = []
        #tracker = [[]]*N doesn't work because of mutability
        tracker = [[] for i in range(N)]
        current_resid = -9999
        current_atoms = []
        for i1, (atom1, res, resid) in enumerate(p_atom_names):
            assert atom1 in connectivity[res]
            for atom2, i2 in current_atoms:
                if resid !=current_resid:# and atom2 == 'C':
                    if atom2 != 'C':
                        continue
                elif not (atom2 in connectivity[res][atom1] and atom1 in connectivity[res][atom2]):
                    continue
             #   if not (atom2 in connectivity[res][atom1] and atom1 in connectivity[res][atom2]):
             #       if resid != current_resid and atom2 == 'C':
             #           current_resid = resid
             #           current_atoms = []
             #       else:
             #           continue
                if atom1=='N' and atom2=='CA':
                    continue
                tracker[i1].append(i2)
                tracker[i2].append(i1)
                if atom_label=='set':
                    if order:
                        names = tuple(sorted((atom_names[i2][0], atom_names[i1][0])))
                    else:
                        names = tuple((atom_names[i2][0], atom_names[i1][0]))
                    bond_types.append(names)
                    bond_idxs.append([i2,i1])
            if resid !=current_resid:# and atom2 == 'C':
                current_resid = resid
                current_atoms = []
            current_atoms.append([atom1,i1])
    if version ==5:
        connectivity = other_parameters['connectivity']
        bond_types = []
        bond_idxs = []
        tracker = [[] for i in range(N)]
        current_resid = -9999
        current_atoms = []
        previous_atoms = []
        for i1, (atom1, res, resid) in enumerate(p_atom_names):
            assert atom1 in connectivity[res]
            if resid!=current_resid:
                previous_atoms = deepcopy(current_atoms)
                current_atoms = []
                current_resid=resid
            if atom1=='N':
                for atom2, i2 in previous_atoms:
                    if atom2=='C':
                        tracker[i1].append(i2)
                        tracker[i2].append(i1)
                        bond_types.append(tuple((atom_names[i2][0], atom_names[i1][0])))
                        bond_idxs.append([i2,i1])

            for atom2, i2 in current_atoms:
                if atom2 in connectivity[res][atom1] and atom1 in connectivity[res][atom2]:
                    tracker[i1].append(i2)
                    tracker[i2].append(i1)
                    names = tuple((atom_names[i2][0], atom_names[i1][0]))
                    bond_types.append(names)
                    bond_idxs.append([i2,i1])
            current_atoms.append([atom1,i1])
    if version <4:
        all_bond_idxs = np.sort(bond_idxs)

        bond_types = []
        bond_idxs = []
        tracker = [[]] # this will keep track of some of the bonds to help work out the angles
        atom1=0
        atom2=1
        counter = 0 #index of the distance N,N+1
        for bond in all_bond_idxs:
            if bond < counter+(N-atom1-1):
                atom2 = atom1+bond-counter+1 # 0-0+1
                tracker[-1].append(atom2) #
            while bond > counter+(N-atom1-2):
                counter+=(N-atom1-1)
                atom1 +=1
                tracker.append([])
                if bond < counter+(N-atom1-1):
                    atom2 = atom1+bond-counter+1
                    tracker[-1].append(atom2)
            if atom_label=='string': #string of atom labels, doesn't handle Proline alternate ordering
                bond_types.append(atom_names[atom1][0]+'_'+atom_names[atom2][0])
                bond_idxs.append([atom1, atom2])
            elif atom_label=='set': #set of atom labels
                if order:
                    names = tuple(sorted((atom_names[atom1][0], atom_names[atom2][0])))
                else:
                    names = (atom_names[atom1][0], atom_names[atom2][0])
                bond_types.append(names)
                bond_idxs.append([atom1, atom2])

    while len(tracker)<N:
        tracker.append([]) #ensure so the next bit doesn't break by indexing N-1

    ##################### Angles/1-3 #####################
    print('Determining angles')
    angle_types = []
    angle_idxs = []

    torsion_types = []
    torsion_idxs = []

    bond_14_idxs = []

    counter=0
    #add missing bonds (each bond counted twice after but atom3>atom1 prevents duplicates later )
    if version < 4:
        for atom1, atom1_bonds in enumerate(deepcopy(tracker)): # for _, [] in enum [[]]
            for atom2 in atom1_bonds:                           # for _ in []
                tracker[atom2].append(atom1)
    # find every angle and add it
    for atom1, atom1_bonds in enumerate(tracker):
        for atom2 in atom1_bonds:
            for atom3 in tracker[atom2]:
                if atom3>atom1: #each angle will only be counter once
                    if order:
                        sort13 = sorted([ (atom_names[atom1][0], atom1), (atom_names[atom3][0], atom3) ], key=lambda x: x[0])
                        names = tuple( (sort13[0][0], atom_names[atom2][0], sort13[1][0]) )

                        angle_types.append(names)
                        angle_idxs.append([sort13[0][1], atom2, sort13[1][1]])
                    else:
                        angle_types.append((atom_names[atom1][0], atom_names[atom2][0],
                                            atom_names[atom3][0]))
                        angle_idxs.append([atom1, atom2, atom3])
                if atom3 != atom1:
                    for atom4 in tracker[atom3]:
                        if atom4>atom1 and atom2!=atom4:# each torsion will be counter once
                            #torsions are done based on the 2 3 atoms, so sort 23
                            if order:
                                sort23 = sorted([ (atom_names[atom2][0], atom2, atom_names[atom1][0], atom1),
                                                  (atom_names[atom3][0], atom3, atom_names[atom4][0], atom4) ], key=lambda x: x[0])
                                names = tuple( (sort23[0][2], sort23[0][0], sort23[1][0], sort23[1][2]) )
                                torsion_types.append(names)
                                torsion_idxs.append([sort23[0][3], sort23[0][1], sort23[1][1], sort23[1][3]])
                            else:
                                torsion_types.append((atom_names[atom1][0], atom_names[atom2][0],
                                                      atom_names[atom3][0], atom_names[atom4][0]))
                                torsion_idxs.append([atom1, atom2, atom3, atom4])
                            bond_14_idxs.append([atom1,atom4])
    #currently have bond_types, angle_types, and torsion_typs + idxs
    bond_idxs = np.array(bond_idxs)
    angle_idxs = np.array(angle_idxs)
    torsion_idxs = np.array(torsion_idxs)
    bond_max_conv = (bond_idxs.max(axis=1)-bond_idxs.min(axis=1)).max()+1
    if bond_max_conv<3 and fix_slice_method:
        bond_max_conv=3
    angle_max_conv = (angle_idxs.max(axis=1)-angle_idxs.min(axis=1)).max()+1
    if angle_max_conv<5 and fix_slice_method:
        angle_max_conv=5
    torsion_max_conv = (torsion_idxs.max(axis=1)-torsion_idxs.min(axis=1)).max()+1
    if torsion_max_conv<7 and fix_slice_method:
        torsion_max_conv=7
        #there is a problem where i accidentally index [padding-3] so if (len -4) < 3 we index -1 which breaks things
        #it shouldn't affect anything to say the max conv is greater than 6
    #this little bit just turns the 'types' list into equivalent parameters
    #key error if you don't have the parameter
    bond_para= np.array([ [bond_equil[bond], bond_force[bond]] if bond in bond_equil
              else [bond_equil[(bond[1],bond[0])], bond_force[(bond[1], bond[0])]]
              for bond in bond_types])
    angle_para=np.array([ [angle_equil[angle], angle_force[angle]] if angle in angle_equil
              else [angle_equil[(angle[2],angle[1],angle[0])], angle_force[(angle[2], angle[1], angle[0])]]
              for angle in angle_types])
    torsion_para=[]
    t_unique=list(set(torsion_types))
    t_unique_para = {}
    max_para = 0
    for torsion in t_unique:
        torsion_b = (torsion[3],torsion[2],torsion[1],torsion[0])
        torsion_xx  = ('X', torsion[2], torsion[1], 'X')
        torsion_xb = ('X', torsion[1], torsion[2], 'X')
        if torsion in torsion_barrier:
            max_para = max(max_para, len(torsion_barrier[torsion]))
            t_unique_para[torsion]= [torsion_factor[torsion], torsion_barrier[torsion],
                                     torsion_phase[torsion],  torsion_period[torsion]]
        elif torsion_b in torsion_barrier:
            max_para = max(max_para, len(torsion_barrier[torsion_b]))
            t_unique_para[torsion]= [torsion_factor[torsion_b], torsion_barrier[torsion_b],
                                     torsion_phase[torsion_b],  torsion_period[torsion_b]]
        elif torsion_xx in torsion_barrier:
            max_para = max(max_para, len(torsion_barrier[torsion_xx]))
            t_unique_para[torsion]= [torsion_factor[torsion_xx], torsion_barrier[torsion_xx],
                                     torsion_phase[torsion_xx],  torsion_period[torsion_xx]]
        elif torsion_xb in torsion_barrier:
            max_para = max(max_para, len(torsion_barrier[torsion_xb]))
            t_unique_para[torsion]= [torsion_factor[torsion_xb], torsion_barrier[torsion_xb],
                                     torsion_phase[torsion_xb],  torsion_period[torsion_xb]]
        else:
            print('ERROR: Torsion %s cannot be found in torsion_barrier and will not be included'% torsion)
    torsion_para = np.zeros((len(torsion_types),4,max_para))
    #we don't want barrier/factor to return nan so set factor to 1 by default
    torsion_para[:,0,:]=1.0
    for i,torsion in enumerate(torsion_types):
        para = t_unique_para[torsion]
        torsion_para[i,:,:len(para[0])]=para
    ##### make phase positive #####
    if absolute_torsion_period:
        torsion_para[:,3,:] = np.abs(torsion_para[:,3,:])



    ###############################  bonds  #################################

    bond_masks = np.zeros((bond_max_conv-1, N-(bond_max_conv-1) + 2*(bond_max_conv-2)),dtype=np.bool)
    bond_conv = (bond_idxs.max(axis=1)-bond_idxs.min(axis=1))-1

    bond_weights = []
    b_equil = np.zeros(bond_masks.shape)
    b_force = np.zeros(bond_masks.shape)
    for i in range(bond_max_conv-1):
        weight = [0.0]*bond_max_conv
        weight[0]=1.0
        weight[i+1]=-1.0
        bond_weights.append(weight)
        mask_index=bond_idxs.min(axis=1)[bond_conv==i]+bond_max_conv-2
        bond_masks[i, mask_index]=True
        b_equil[i, mask_index]=bond_para[bond_conv==i,0]
        b_force[i, mask_index]     =bond_para[bond_conv==i,1]

    ###############################  angles  #################################

    angle_conv = (angle_idxs-angle_idxs.min(axis=1).reshape(-1,1))#relative positions of atoms
    angle_conv = np.where((angle_conv[:,0]<angle_conv[:,2]).reshape(-1,1),angle_conv,angle_conv[:,[2,1,0]]) #remove mirrors
    angle_unique = np.unique(angle_conv, axis=0) #unique

    angle_masks = np.zeros((len(angle_unique), N-(angle_max_conv-1)+2*(angle_max_conv-3)),dtype=np.bool)
    angle_weights=[]
    a_equil = np.zeros(angle_masks.shape)
    a_force = np.zeros(angle_masks.shape)
    for i, angle in enumerate(angle_unique):
        weight = [[0.0]*angle_max_conv,[0.0]*angle_max_conv] # 2xsize
        weight[0][angle[0]]=1.0
        weight[0][angle[1]]=-1.0
        weight[1][angle[1]]=-1.0
        weight[1][angle[2]]=1.0
        angle_weights.append(weight)
        mask_index = angle_idxs.min(axis=1)[(angle_conv==angle).all(axis=1)]+angle_max_conv-3
        a_equil[i, mask_index]=angle_para[(angle_conv==angle).all(axis=1),0]
        a_force[i, mask_index]     =angle_para[(angle_conv==angle).all(axis=1),1]
        angle_masks[i, mask_index]=True


    ###############################  torsion  #################################

    torsion_conv = (torsion_idxs-torsion_idxs.min(axis=1).reshape(-1,1))#relative positions of atoms
    torsion_conv = np.where((torsion_conv[:,0]<torsion_conv[:,3]).reshape(-1,1),torsion_conv,torsion_conv[:,[3,2,1,0]]) #remove mirrors
    torsion_unique = np.unique(torsion_conv, axis=0) #unique

    torsion_masks = np.zeros((len(torsion_unique), N-(torsion_max_conv-1)+2*(torsion_max_conv-4)),dtype=np.bool)
    torsion_weights=[]
    ts = torsion_masks.shape
    t_para   = np.zeros((torsion_masks.shape[0],torsion_masks.shape[1],
                        torsion_para.shape[1], torsion_para.shape[2]))
    #we don't want barrier/factor to return nan so set factor to 1 by default
    t_para[:,:,0,:]=1.0
    for i, torsion in enumerate(torsion_unique):
        weight = [[0.0]*torsion_max_conv, [0.0]*torsion_max_conv, [0.0]*torsion_max_conv]
        weight[0][torsion[0]] =  1.0 #b1 = ri-rj
        weight[0][torsion[1]] = -1.0
        weight[1][torsion[1]] =  1.0 #b2 = rj-rk
        weight[1][torsion[2]] = -1.0
        weight[2][torsion[2]] = -1.0 #b3 = rl-rk
        weight[2][torsion[3]] =  1.0
        torsion_weights.append(weight)
        mask_index = torsion_idxs.min(axis=1)[(torsion_conv==torsion).all(axis=1)]+torsion_max_conv-4
        torsion_masks[i,mask_index]=True
        t_para[i,mask_index] = torsion_para[(torsion_conv==torsion).all(axis=1)]

    if NB=='matrix':
        #cdist is easier to work with than pdist, batch pdist was removed from torch and has not been readded as of writting this
        vdw_R = 0.5*torch.cdist(atom_R.view(-1,1), -atom_R.view(-1, 1)).triu(diagonal=1)
        vdw_e = (atom_e.view(1,-1)*atom_e.view(-1, 1)).triu(diagonal=1).sqrt()
        #set 1-2, and 1-3 distances to 0.0
        vdw_R[bond_idxs.T]=0.0
        vdw_e[bond_idxs.T]=0.0
        vdw_R[angle_idxs[:,(0,2)].T]=0.0
        vdw_e[angle_idxs[:,(0,2)].T]=0.0
        vdw_R[torsion_idxs[:,(0,3)].T]=0.0
        vdw_e[torsion_idxs[:,(0,3)].T]=0.0

        e_=permitivity #permitivity
        atom_charges=torch.tensor(atom_charges)
        q1q2=(atom_charges.view(1,-1)*atom_charges.view(-1,1)/e_).triu(diagonal=1) #Aij=bi*bj
        q1q2[bond_idxs.T]=0.0
        q1q2[angle_idxs[:,(0,2)].T]=0.0
        q1q2[torsion_idxs[:,(0,3)].T]=0.0

        #1-4 are should be included but scaled
        bond_14_idxs = np.array(bond_14_idxs)
        vdw_14R=0.5*(atom_R[bond_14_idxs[:,0]]+atom_R[bond_14_idxs[:,1]])
        vdw_14e=(atom_e[bond_14_idxs[:,0]]+atom_e[bond_14_idxs[:,1]]).sqrt()
        q1q2_14=(atom_charges[bond_14_idxs[:,0]]*atom_charges[bond_14_idxs[:,1]])/e_

        return (bond_masks,  b_equil, b_force, bond_weights,
                angle_masks, a_equil, a_force, angle_weights,
                torsion_masks, t_para, torsion_weights,
                vdw_R, vdw_e, vdw_14R, vdw_14e,
                q1q2, q1q2_14
               )



    return (bond_masks,  b_equil, b_force, bond_weights,
            angle_masks, a_equil, a_force, angle_weights,
            torsion_masks, t_para, torsion_weights)


def get_conv_pad_res(dataset, pdb_atom_names,
                              absolute_torsion_period=True,
                              NB=('matrix',)[0],
                              fix_terminal=True,
                              fix_charmm_residues=True,
                              correct_1_4=True,
                              permitivity=1.0
                             ):
    '''
    ##INPUTS##

    dataset:         one frame of a trajectory of shape [3, N]

    pdb_atom_names:  should be an array of shape [N,2]
                     pdb_atom_names[:,0] is the pdb_atom_names and
                     pdb_atom_names[:,1] is the residue names


    '''

    if len(dataset.shape)!=3:
        raise Exception('967 dataset frame here should be of shape [R, M, 3] not %s'%str(dataset.shape))
    if dataset.shape != pdb_atom_names.shape:
        raise Exception('969 dataset.shape != pdb_atom_names.shape')


    #get amber parameters
    (amber_atoms, atom_mass, atom_polarizability, bond_force, bond_equil,
    angle_force, angle_equil, torsion_factor, torsion_barrier, torsion_phase,
    torsion_period, improper_factor, improper_barrier, improper_phase,
    improper_period, other_parameters) =  get_amber_parameters()


    R, M, D = dataset.shape # N residues, Max atom per res, dimension D
    N = R*M
    # [R*M, 3], 3 = atom, res, resid
    pdb_atom_names = pdb_atom_names.reshape(-1,3)
    # [R*M, 3], 3 = x, y, z
    dataset = dataset.reshape(-1,3)

    if fix_terminal:#fix atoms and residues not in amber parameters
        pdb_atom_names[pdb_atom_names[:, 0]=='OXT',0]='O'
    if fix_charmm_residues:
        pdb_atom_names[pdb_atom_names[:, 1]=='HSD',0]='HID'
        pdb_atom_names[pdb_atom_names[:, 1]=='HSE',0]='HIE'

    #pdb atoms -> amber atom names and residues
    padded_atom_names = np.array([[amber_atoms[res][atom],res, resid] if atom is not None else [atom, res, resid] for atom, res, resid in pdb_atom_names])
    unpadded_atom_names = [[amber_atoms[res][atom],res, resid] for atom, res, resid in pdb_atom_names if atom is not None]
    padded_atom_charges = np.array([other_parameters['charge'][res][atom] if atom is not None else np.nan for atom, res, _ in padded_atom_names])
    if padded_atom_names.shape != dataset.shape: # just a little check
        raise Exception('996 padded_atom_names!=dataset.shape')
    atom_names = padded_atom_names
    atom_charges = padded_atom_charges
    print('Determining bonds')


    connect = other_parameters['connectivity']
    #connectivity = [[]]*N # careful with mutability
    connectivity = [[] for i in range(N)]
    current_resid = -9999
    current_atoms = []
    for i1, (atom1, res, resid) in enumerate(padded_atom_names):
        if atom1 is None:
            continue
        if resid!= current_resid:
            current_resid = resid
            current_atoms = []
        assert atom1 in connect[res]
        for atom2, i2 in current_atoms:
            if atom2 in connect[res][atom1] and atom1 in connect[res][atom2]:
                connectivity[i1].append(i2)
                connectivity[i2].append(i1)
 #   cmat = torch.cdist(dataset, dataset) #[R*M,3 ]-> [R*M, R*M]
 #   #1.643 was max bond distance in MurD test, 2.129 was the smallest nonbonded distance
 #   #can't say what the best solution is but somewhere in the middle will probably be okay
 #   all_bond_mask = (cmat<(1.643+2.1269)/2).triu(diagonal=1) # [R*M,R*M]
 #   bond_idxs = all_bond_mask.nonzero() # [B x 2]
 #   #name_set = set(atom_names[:,0])
#
#    connectivity = [[] for i in range(N)] # this will keep track of some of the bonds to help work out the angles
#    for i,j in bond_idxs:
#        connectivity[i].append(j)
#        connectivity[j].append(i)
#    ##################### Angles/1-3 #####################
    print('Determining angles')

    bond_idxs_ = []
    angle_idxs = []
    torsion_idxs = []
    bond_14_idxs = []

    bond_para = []
    angle_para = []
    torsion_para_ = []

    for atom1, atom2_list in enumerate(connectivity):
        for atom2 in atom2_list:
            a1, a2 = atom_names[atom1][0], atom_names[atom2][0]
            if atom1 < atom2: #stops any pair of atoms being selected twice
                bond_idxs_.append([atom1, atom2])
                for b in [(a1,a2), (a2,a1)]:
                    if b in bond_equil:
                        bond_para.append([bond_equil[b], bond_force[b]])
                        break # break prevents any bond from beind added twice
                else:
                    raise Exception('No associated bond parameter')

            for atom3 in connectivity[atom2]:
                a3 = atom_names[atom3][0]
                if atom3 > atom1: #each angle will only be counter once
                    angle_idxs.append([atom1,atom2,atom3])
                    for a in [(a1,a2,a3), (a3,a2,a1)]:
                        if a in angle_equil:
                            angle_para.append([angle_equil[a], angle_force[a]])
                            break
                    else:
                        raise Exception('No associated angle parameter')
                if atom3 != atom1: #don't go back to same atom
                    for atom4 in connectivity[atom3]:
                        if atom4 > atom1 and atom2!=atom4:
                            torsion_idxs.append([atom1, atom2, atom3, atom4])
                            bond_14_idxs.append([atom1, atom4])
                            a4 = atom_names[atom4][0]
                            for t in [(a1,a2,a3,a4),(a4,a3,a2,a1),('X',a2,a3,'X'),('X',a3,a2,'X')]:
                                if t in torsion_barrier:
                                    torsion_para_.append(torch.tensor([
                                            torsion_factor[t],
                                            torsion_barrier[t],
                                            torsion_phase[t],
                                            torsion_period[t]]))
                                    break #each torsion only counter once
                            else:
                                raise Exception('No associated torsion parameter')

    bond_idxs_ = torch.as_tensor(bond_idxs_)
    angle_idxs = torch.tensor(angle_idxs)
    torsion_idxs = torch.tensor(torsion_idxs)
    bond_14_idxs = torch.tensor(bond_14_idxs)
    bond_para = torch.tensor(bond_para)
    angle_para = torch.tensor(angle_para)
    max_number_torsion_para = max([tf.shape[1] for tf in torsion_para_])
    torsion_para = torch.zeros(torsion_idxs.shape[0],4,max_number_torsion_para)
    torsion_para[:,0,:]=1.0
    for i,tf in enumerate(torsion_para_):
        torsion_para[i,:,0:tf.shape[1]]=tf
    if absolute_torsion_period:
        torsion_para[:,3,:] = np.abs(torsion_para[:,3,:])
    ###### Gather based potential ######
    #currently for data [B, R*M, 3] or [B, 3, N]
    aij0 = bond_idxs.reshape(-1,2,1).eq(angle_idxs[:,(0,1)].view(-1,2,1).permute(2,1,0)).all(dim=1)
    aij1 = bond_idxs.reshape(-1,2,1).eq(angle_idxs[:,(1,0)].view(-1,2,1).permute(2,1,0)).all(dim=1)
    ajk0 = bond_idxs.reshape(-1,2,1).eq(angle_idxs[:,(1,2)].view(-1,2,1).permute(2,1,0)).all(dim=1)
    ajk1 = bond_idxs.reshape(-1,2,1).eq(angle_idxs[:,(2,1)].view(-1,2,1).permute(2,1,0)).all(dim=1)
    ij_jk = torch.stack([torch.where((aij0+aij1).T)[1], torch.where((ajk0+ajk1).T)[1]])
    aij_ = aij1.float()-aij0.float() #sign change needed for loss_function equation
    ajk_ = ajk0.float()-ajk1.float()
    angle_mask = torch.stack([aij_.sum(dim=0), ajk_.sum(dim=0)])

    #following are [N_bonds, N_torsions] arrays comparing if the ij or jk are the same
    ij0 = bond_idxs.reshape(-1,2,1).eq(torsion_idxs[:,(0,1)].view(-1,2,1).permute(2,1,0)).all(dim=1)
    ij1 = bond_idxs.reshape(-1,2,1).eq(torsion_idxs[:,(1,0)].view(-1,2,1).permute(2,1,0)).all(dim=1)
    jk0 = bond_idxs.reshape(-1,2,1).eq(torsion_idxs[:,(1,2)].view(-1,2,1).permute(2,1,0)).all(dim=1)
    jk1 = bond_idxs.reshape(-1,2,1).eq(torsion_idxs[:,(2,1)].view(-1,2,1).permute(2,1,0)).all(dim=1)
    kl0 = bond_idxs.reshape(-1,2,1).eq(torsion_idxs[:,(2,3)].view(-1,2,1).permute(2,1,0)).all(dim=1)
    kl1 = bond_idxs.reshape(-1,2,1).eq(torsion_idxs[:,(3,2)].view(-1,2,1).permute(2,1,0)).all(dim=1)
    ij_jk_kl = torch.stack([torch.where((ij0+ij1).T)[1],
                            torch.where((jk0+jk1).T)[1],
                            torch.where((kl0+kl1).T)[1]])
    ij_ = ij0.float()-ij1.float()
    jk_ = jk0.float()-jk1.float()
    kl_ = kl0.float()-kl1.float()
    torsion_mask = torch.stack([ij_.sum(dim=0), jk_.sum(dim=0), kl_.sum(dim=0)])

    #j-i i->j
    #i-j j->i reverse
    #k-j j->k
    #j-k k->j reverse
    #l-k k->l
    #k-l l->k reverse


    if NB=='matrix':
        equiv_t=other_parameters['equivalences']
        vdw_para = other_parameters['vdw_potential_well_depth']
        #switch these around so that values point to key
        equiv = {}
        for i in equiv_t.keys():
            j = equiv_t[i]
            for k in j:
                equiv[k]=i
        atom_R = torch.tensor([vdw_para[equiv.get(i,i)][0] if i is not None else np.nan for i, j, k in atom_names]) #radius
        atom_e = torch.tensor([vdw_para[equiv.get(i,i)][1] if i is not None else np.nan for i, j, k in atom_names]) #welldepth
        #cdist is easier to work with than pdist, batch pdist doesn't seem to exist too
        vdw_R = 0.5*torch.cdist(atom_R.view(-1,1), -atom_R.view(-1, 1)).triu(diagonal=1)
        vdw_e = (atom_e.view(1,-1)*atom_e.view(-1, 1)).triu(diagonal=1).sqrt()
        #set 1-2, and 1-3 distances to 0.0
        vdw_R[list(bond_idxs.T)]=0.0
        vdw_e[list(bond_idxs.T)]=0.0
        vdw_R[list(angle_idxs[:,(0,2)].T)]=0.0
        vdw_e[list(angle_idxs[:,(0,2)].T)]=0.0
        if correct_1_4:
            # sum A/R**12 - B/R**6; A = e* (R**12); B = 2*e *(R**6)
            # therefore scale vdw by setting e /= 2.0
            #vdw_R[list(torsion_idxs[:,(0,3)].T)]/=2.0
            vdw_e[list(torsion_idxs[:,(0,3)].T)]/=2.0
        else:
            vdw_R[list(torsion_idxs[:,(0,3)].T)]=0.0
            vdw_e[list(torsion_idxs[:,(0,3)].T)]=0.0
        vdw_R[torch.isnan(vdw_R)]=0.0
        vdw_e[torch.isnan(vdw_e)]=0.0

        #partial charges are given as fragments of electron charge.
        #Can convert coulomb energy into kcal/mol by multiplying with 332.05.
        #therofore multiply q by sqrt(332.05)=18.22
        e_=permitivity #permittivity
        atom_charges=torch.tensor(atom_charges)
        q1q2=(atom_charges.view(1,-1)*atom_charges.view(-1,1)/e_).triu(diagonal=1) #Aij=bi*bj
        q1q2[list(bond_idxs.T)]=0.0
        q1q2[list(angle_idxs[:,(0,2)].T)]=0.0
        if correct_1_4:
            q1q2[list(torsion_idxs[:,(0,3)].T)]/=1.2
        else:
            q1q2[list(torsion_idxs[:,(0,3)].T)]=0.0
        #1-4 are should be included but scaled
        return (
                bond_idxs, bond_para,
                angle_idxs, angle_para, angle_mask, ij_jk,
                torsion_idxs, torsion_para, torsion_mask, ij_jk_kl,
                vdw_R, vdw_e,
                q1q2,
               )
    return (
            bond_idxs, bond_para,
            angle_idxs, angle_para, angle_mask, ij_jk,
            torsion_idxs, torsion_para, torsion_mask, ij_jk_kl,
           )

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.abspath('../'))
    import biobox


