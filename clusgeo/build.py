import ase
from clusgeo.cluster import ClusGeo
import clusgeo
import numpy as np
import ase, ase.io

from ase.cluster.icosahedron import Icosahedron
from ase.cluster.octahedron import Octahedron
from ase.cluster import wulff_construction
from scipy.spatial.distance import cdist, pdist
from scipy.spatial.distance import squareform


import random
import time
import dscribe
from dscribe.descriptors import SOAP
from dscribe import utils
from ase.visualize import view
import copy

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

# helper functions
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


def get_scaffold(shape = "ico", i = 3, latticeconstant = 3.0,
	energies = [0.5,0.4,0.3], surfaces = [(1, 0, 0), (1, 1, 1), (1, 1, 0)]):
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
    return ClusGeo(atoms)



class Clusterer:

    def __init__(self, bondmatrix, positions, ntypeB, eAA, eAB, eBB, com=None, coreEnergies=[0,0]):

        self.bondmat = bondmatrix
        self.positions = positions
        self.ntypeB = ntypeB

        # list of possible atom types
        # self.types = types


        # this will be used as center of the cluster
        if com == None:
            # WARNING: this is the geometric center, not the actual center of mass!
            self.com = np.mean(positions)
        else:
            self.com = com

        self.coreEnergies = np.asarray(coreEnergies)

        tmp = np.zeros((2,2))
        tmp[0,0] = eAA
        tmp[0,1] = eAB
        tmp[1,0] = eAB
        tmp[1,1] = eBB
        self.nearEnergies = tmp

        self.atomTypes = np.zeros(positions.shape[0], dtype=np.int32)
        self.atomTypes[0:self.ntypeB] = 1
        np.random.shuffle(self.atomTypes)


        # compute the coreness of each atom
        self.coreness = np.zeros(positions.shape[0])
        for i in range(positions.shape[0]):
            self.coreness[i] = np.linalg.norm(self.com - positions[i])
        maxdist = np.max(self.coreness)

        self.coreness /= maxdist
        self.coreness = 1 - self.coreness
        self.coreness = 2*self.coreness - 1
    # --- end of init --- #

    def Evolve(self, kT, nsteps):

        energyBefore = np.sum(self.coreEnergies[self.atomTypes] * self.coreness) # corification contribution
        nearenergymat = self.nearEnergies[self.atomTypes][:,self.atomTypes]
        energyBefore += np.sum(np.multiply(self.bondmat,nearenergymat))
        timeBef = time.time()
        for s in range(nsteps):
            tmp = np.copy(self.atomTypes)
            idx = np.random.choice(self.atomTypes.shape[0], 2, replace=False)
            tmp[idx] = self.atomTypes[np.flipud(idx)]
            energyAfter = np.sum(self.coreEnergies[tmp] * self.coreness)

            nearenergymat = self.nearEnergies[tmp][:, tmp]
            energyAfter += np.sum(np.multiply(self.bondmat,nearenergymat))

            if random.random() < np.exp(-(energyAfter-energyBefore)/kT):
                energyBefore = energyAfter
                self.atomTypes = tmp

        timeAft = time.time()
#        print(timeAft - timeBef)
    # --- end of Evolve --- #


    def Reset(self):
        self.atomTypes = np.zeros(self.positions.shape[0], dtype=np.int32)
        self.atomTypes[0:self.ntypeB] = 1
        np.random.shuffle(self.atomTypes)


    # --- end of Reset --- #
def get_unique_clusters(eAA,eAB,eBB,cEA,cEB,typeA, typeB, ntypeB, n_clus = 1, clusSize=3,clusShape="ico"):
    atoms = get_scaffold(shape = clusShape, i = clusSize)
    bondmatrix = _get_connectivity(atoms)
    desc = SOAP([typeA,typeB],6.0,6,6,sparse=False, periodic=False,average=True,crossover=True)
    final_atom_list = []
    
    positions = atoms.get_positions()
    print(positions)
    coreEnergies = [ cEA, cEB ]
    
    atomList = []
    for i in range(0,howMany*2):
        cluster = Clusterer(bondmatrix, positions, ntypeB, eAA, eAB, eBB, com=None, coreEnergies=coreEnergies)
    
        kT = 0.2
        nsteps = 1000
    
        cluster.Evolve(kT, nsteps)
        actual_types = cluster.atomTypes.copy()
        actual_types[actual_types == 0] = typeA
        actual_types[actual_types == 1] = typeB
    
        atoms.set_atomic_numbers(actual_types)
        atomsList.append(atoms.copy())
    
    x = utils.batch_create(desc, atomList,1 ,  positions=None, create_func=None, verbose=True)
    ranks = clusgeo.cluster._rank_fps(x, K = None, greedy =False, is_safe = True)
    for i in range(0,howMany):
        view(atomsList[ranks[i]])
        final_atoms_list.append(atomsList[ranks[i]])

    cluster.Reset()
    return final_atoms_list

def get_segregated(typeA, typeB, ntypeB, n_clus = 1, clusSize=3,clusShape="ico"):
    return get_unique_clusters(-1,1,-1,0,0,typeA,typeB,ntypeB,howMany, clusSize, clusShape)
def get_core_shell(typeA, typeB, ntypeB, n_clus = 1, clusSize=3,clusShape="ico"):
    return get_unique_clusters(0,0,0,1,0,typeA,typeB,ntypeB,howMany, clusSize, clusShape)
def get_random(typeA, typeB, ntypeB, n_clus = 1, clusSize=3,clusShape="ico"):
    return get_unique_clusters(0,0,0,0,0,typeA,typeB,ntypeB,howMany, clusSize, clusShape)
def get_ordered(typeA, typeB, ntypeB, n_clus = 1, clusSize=3,clusShape="ico"):
    return get_unique_clusters(1,-1,1,0,0,typeA,typeB,ntypeB,howMany, clusSize, clusShape)




# types: list of atomic types

# stoichiometry: list of int

# phase-type (optional):
# core-shell
# ordered alloy
# random alloy
# segregated alloy
# single-atom alloy



# in utils ?
def get_unique_clusters_in_range(

    ):
	pass


if __name__=="__main__":
   x = get_unique_ord(29,49,30,5,clusSize=4)

