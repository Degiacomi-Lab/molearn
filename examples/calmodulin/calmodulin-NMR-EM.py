from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import numpy as np
import pandas as pd

source_dir = './curated'
destination_dir = './minimized'

# Create destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

for pdb_file in os.listdir(source_dir):
    if pdb_file.endswith('.pdb'):
        pdb_path = os.path.join(source_dir, pdb_file)

        # Setup the simulation
        print(f'Loading {pdb_file}...')
        pdb = PDBFile(pdb_path)
        forcefield = ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')
        modeller = Modeller(pdb.topology, pdb.positions)

        # Adding hydrogens and solvent
        print('Adding hydrogens...')
        modeller.addHydrogens(forcefield)
        print('Adding solvent...')
        modeller.addSolvent(forcefield, model='tip3p', padding=1 * nanometer)

        # Start energy minimization
        print('Minimizing...')
        system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME)
        integrator = VerletIntegrator(0.001 * picoseconds)
        simulation = Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)

        # Minimize energy (equivalent to emtol and nsteps in GROMACS)
        simulation.minimizeEnergy(tolerance=Quantity(value=10, unit=kilojoule / (nanometer * mole)), maxIterations=0)

        # Save minimized structure to the destination directory
        print(f'Saving minimized {pdb_file}...')
        minimized_pdb_file = os.path.join(destination_dir, f'minimized_{pdb_file}')
        state = simulation.context.getState(getPositions=True)
        with open(minimized_pdb_file, 'w') as output:
            PDBFile.writeFile(simulation.topology, state.getPositions(), output)

        print(f'Done with {pdb_file}\n')

