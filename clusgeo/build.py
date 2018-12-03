import ase

def get_scaffold()
import numpy as np
import ase, ase.io

from ase.cluster.icosahedron import Icosahedron
from ase.cluster.octahedron import Octahedron
from ase.cluster import wulff_construction
from scipy.spatial.distance import cdist, pdist
from scipy.spatial.distance import squareform

### GLOBAL ###

#The following are BCC/HCP lattice constants taken from 
#http://periodictable.com/Properties/A/LatticeConstants.html 
#They are multiplied by a factor of √2, since ASE will divide the 
#constant by √2 to get the distance between FCC (111) planes.

tm_dict = {'Sc': 4.6796, 'Ti': 4.1731, 'V': 4.2851, 'Cr': 4.1154, 'Mn': 1.2604, 'Fe': 4.0538, 
           'Co': 3.5456, 'Zn': 3.7687, 'Y': 5.1582, 'Zr': 4.5707, 'Nb': 4.6675, 'Mo': 4.4505, 
           'Tc': 3.8679, 'Ru': 3.8267, 'Cd': 4.2135, 'Hf': 4.5204, 'Ta': 4.6687, 'W': 4.4763, 
           'Re': 3.9046, 'Os': 3.8670, 'Hg': 4.2497}


### DEFINE ###

def get_scaffold(shape = "ico", i = 3, latticeconstant = 3.0,
	energies = [0.5,0.4,0.3], surfaces = [(1, 0, 0),
                      (1, 1, 1),
                      (1, 1, 0)]):
    if shape == "ico":
        atoms = Icosahedron('X', noshells = i, latticeconstant = latticeconstant)
    elif shape == "octa":
        atoms = Octahedron('X', length = i, latticeconstant = latticeconstant)
    elif shape == "wulff":
        # i gives size in atoms
        atoms = wulff_construction('X',
            latticeconstant = latticeconstant,
            surfaces=surfaces,
            energies=energies,
            size=i,
            structure='fcc',
            rounding='above')
    else:
        raise NameError("shape argument unknown! Use ico, octa or wulff")
    return atoms

def _get_distances_to_com(atoms):
    center_of_mass = atoms.get_center_of_mass()
    distances = cdist(atoms.get_positions(), center_of_mass.reshape((-1,3)))
    print(distances.shape)
    return distances


def _get_connectivity(atoms):
    dmat = pdist(atoms.get_positions())
    min_pos = dmat.min()
    dmat = squareform(dmat)

    connectivity_matrix = np.isclose(dmat, 
        np.ones(np.shape(dmat)) * min_pos,
        rtol=1e-01)

    return connectivity_matrix





# types: list of atomic types

# stoichiometry: list of int

# phase-type (optional):
# core-shell
# ordered alloy
# random alloy
# segregated alloy
# single-atom alloy







def get_binary_configuration():
	pass


# convenience functions calling 
def create_coreshell():
	pass

def create_ordered():
	pass


def create_random():
	pass

def create_segragated():
	pass

def create_saa():
	pass


def get_dissimilar_configurations():
	pass



