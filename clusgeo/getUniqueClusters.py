import random
import numpy as np
import scaffold
import time
import dscribe
from dscribe.descriptors import SOAP, ACSF
import ase
from ase import io
from dscribe import utils
import clusgeo.surface, clusgeo.environment
from ase.visualize import view
import copy


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
def get_unique_clusters(eAA,eAB,eBB,cEA,cEB,typeA, typeB, ntypeB, howMany, clusSize=3,clusShape="ico"):
    atoms = scaffold.get_scaffold(shape = clusShape, i = clusSize)
    bondmatrix = scaffold.get_connectivity(atoms)
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
        atomList.append(atoms.copy())
    
    x = utils.batch_create(desc, atomList,1 ,  positions=None, create_func=None, verbose=True)
    ranks = clusgeo.environment.rank_sites(x, K = None, idx=[], greedy =False, is_safe = True)
    for i in range(0,howMany):
        view(atomList[ranks[i]])
        final_atom_list.append(atomList[ranks[i]])

    cluster.Reset()
    return final_atom_list
    print("Done")

def get_unique_seg(typeA, typeB, ntypeB, howMany, clusSize=3,clusShape="ico"):
    return get_unique_clusters(-1,1,-1,0,0,typeA,typeB,ntypeB,howMany, clusSize, clusShape)
def get_unique_core(typeA, typeB, ntypeB, howMany, clusSize=3,clusShape="ico"):
    return get_unique_clusters(0,0,0,1,0,typeA,typeB,ntypeB,howMany, clusSize, clusShape)
def get_unique_rand(typeA, typeB, ntypeB, howMany, clusSize=3,clusShape="ico"):
    return get_unique_clusters(0,0,0,0,0,typeA,typeB,ntypeB,howMany, clusSize, clusShape)
def get_unique_ord(typeA, typeB, ntypeB, howMany, clusSize=3,clusShape="ico"):
    return get_unique_clusters(1,-1,1,0,0,typeA,typeB,ntypeB,howMany, clusSize, clusShape)

if __name__=="__main__":
   x = get_unique_ord(29,49,30,5,clusSize=4)
