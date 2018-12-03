import numpy as np
import ase, ase.io

from ase.cluster.icosahedron import Icosahedron
from ase.cluster.octahedron import Octahedron
from ase.cluster import wulff_construction
from ase.visualize import view
from ase.cluster.cubic import FaceCenteredCubic

from scipy.spatial.distance import cdist, pdist
from scipy.spatial.distance import squareform


### DEFINE ###
def _make_atoms(symbol, surfaces, energies, factor, structure, latticeconstant):
    #from ase.utils import basestring
    layers1 = factor * np.array(energies)
    layers = np.round(layers1).astype(int)
    atoms = structure(symbol, surfaces, layers,
                      latticeconstant=latticeconstant)
    print('Created a cluster with %i atoms: %s' % (len(atoms), str(layers)))
    return (atoms, layers)


def get_scaffold(shape = "ico", i = 3, latticeconstant = 3.0):
    if shape == "ico":
        print("shape is", shape)
        atoms = Icosahedron('X', noshells = i, latticeconstant = latticeconstant)
    elif shape == "octa":
        print("shape is", shape)
        atoms = Octahedron('X', length = i, latticeconstant = latticeconstant)
    elif shape == "wulff":
        print("shape is", shape)
        # i gives size in atoms

        atoms = wulff_construction('X',
            latticeconstant = latticeconstant,
            surfaces=[(1, 0, 0),
                      (1, 1, 1),
                      (1, 1, 0)],
            energies=[0.1, 0.5, 0.15],
            size=i,
            structure='fcc',
            rounding='above')

        # i scales the shape proportional to the energies
        #atoms, layers = _make_atoms('X', 
        #    surfaces=[(1, 0, 0),
        #              (1, 1, 1),
        #              (1, 1, 0)],
        #    energies=[0.1, 0.5, 0.15],
        #    factor = i,
        #    structure = FaceCenteredCubic,
        #    latticeconstant = latticeconstant,
        #    )

    else:
        raise NameError("shape argument unknown! Use ico, octa or wulff")
    return atoms

def get_distance_to_com(atoms):
    center_of_mass = atoms.get_center_of_mass()
    distances = cdist(atoms.get_positions(), center_of_mass.reshape((-1,3)))
    print(distances.shape)
    return distances


def get_connectivity(atoms):
    dmat = pdist(atoms.get_positions())
    min_pos = dmat.min()
    dmat = squareform(dmat)

    connectivity_matrix = np.isclose(dmat, 
        np.ones(np.shape(dmat)) * min_pos,
        rtol=1e-01)

    return connectivity_matrix


### INPUT ###

# shape: octa ico wulff

# size: shells

# types: list of atomic types

# stoichiometry: list of int

# phase-type (optional):
# core-shell
# ordered alloy
# random alloy
# segregated alloy
# single-atom alloy


#The following are BCC/HCP lattice constants taken from 
#http://periodictable.com/Properties/A/LatticeConstants.html 
#They are multiplied by a factor of √2, since ASE will divide the 
#constant by √2 to get the distance between FCC (111) planes.

tm_dict = {'Sc': 4.6796, 'Ti': 4.1731, 'V': 4.2851, 'Cr': 4.1154, 'Mn': 1.2604, 'Fe': 4.0538, 
           'Co': 3.5456, 'Zn': 3.7687, 'Y': 5.1582, 'Zr': 4.5707, 'Nb': 4.6675, 'Mo': 4.4505, 
           'Tc': 3.8679, 'Ru': 3.8267, 'Cd': 4.2135, 'Hf': 4.5204, 'Ta': 4.6687, 'W': 4.4763, 
           'Re': 3.9046, 'Os': 3.8670, 'Hg': 4.2497}





### PROCESS ###

if __name__ == '__main__':
    atoms = get_scaffold(shape = "ico", i = 3)

    # get the shell number
    print("tags")
    print(atoms.get_tags())


    # get the distance of each atom to the core 
    # (center of mass = center of atoms)
    distances = get_distance_to_com(atoms)

    print("distances to center of mass")
    print(distances)


    connectivity_matrix = get_connectivity(atoms)
    print("connectivity_matrix")
    print(connectivity_matrix.sum(axis = 1))


    # OUTPUT #

    #ase.io.write("testout_cluster.xyz", atoms)
    ase.io.write("temptest_cluster.pdb", atoms)

    # comment out for quick visualization
    view(atoms)