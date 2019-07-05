import ase
from cluskit.cluster import Cluster
import cluskit
import numpy as np
import ase, ase.io

from ase.cluster.icosahedron import Icosahedron
from ase.cluster.octahedron import Octahedron
from ase.cluster import wulff_construction
from scipy.spatial.distance import cdist, pdist
from scipy.spatial.distance import squareform
from scipy.spatial import Voronoi


import random
import time
import dscribe
from ase.visualize import view
import copy

### GLOBAL ###

#The following are BCC/HCP lattice constants taken from 
#http://periodictable.com/Properties/A/LatticeConstants.html 
#They are multiplied by a factor of sqrt 2, since ASE will divide the 
#constant by sqrt 2 to get the distance between FCC (111) planes.

tm_dict = {'Sc': 4.6796, 'Ti': 4.1731, 'V': 4.2851, 'Cr': 4.1154, 'Mn': 1.2604, 'Fe': 4.0538, 
           'Co': 3.5456, 'Zn': 3.7687, 'Y': 5.1582, 'Zr': 4.5707, 'Nb': 4.6675, 'Mo': 4.4505, 
           'Tc': 3.8679, 'Ru': 3.8267, 'Cd': 4.2135, 'Hf': 4.5204, 'Ta': 4.6687, 'W': 4.4763, 
           'Re': 3.9046, 'Os': 3.8670, 'Hg': 4.2497}

###

# helper functions
def _get_distances_to_com(atoms):
    """Helper function to get the distances to the center of mass

    Args:
        atoms (ase.Atoms) : ase.Atoms object

    Returns:
        1D ndarray :    distances of each atom of the atoms object to
                        center of mass 
    """
    center_of_mass = atoms.get_center_of_mass()
    distances = cdist(atoms.get_positions(), center_of_mass.reshape((-1,3)))
    return distances


def _get_connectivity(positions, max_bondlength = None):
    """Helper function to determine the connectivity between atoms in a crystalline (!)
    nanocluster. Takes an ase atoms object as input.
    Returns the bond matrix / connectivity matrix as 
    squared array of ones (connected) and zeros.

    Args:
        positions (2D ndarray) :    atomic positions of the structure in Angstrom
        max_bondlength (float) :    distance up to which two atoms are considered
                                    bound. If None, the minimum distance is used
                                    as a guess with a tolerance factor of 0.1.
                                    Works well with equidistant atoms.
    Returns:
        2d ndarray :    boolean bond matrix / connectivity matrix with ones
                        indicating the connecting atoms. The diagonal is set to zero
    """
    dmat = pdist(positions)
    min_dist = dmat.min()
    dmat = squareform(dmat)

    if max_bondlength == None:
        bond_matrix = np.isclose(dmat, 
            np.ones(np.shape(dmat)) * min_dist,
            rtol=1e-01)
    else:
        bond_matrix = np.less_equal(dmat, max_bondlength) 

    np.fill_diagonal(bond_matrix, False)

    return bond_matrix

def _get_voronoi_connectivity(positions):
    """Helper function to determine the voronoi connectivity 
    between atoms in a nanocluster. Takes an ase atoms object as input.
    Returns the bond matrix / connectivity matrix as 
    squared array of ones (connected) and zeros.

    Args:
        positions (2D ndarray) :    atomic positions of the structure in Angstrom
        
    Returns:
        2d ndarray :    boolean bond matrix / connectivity matrix with ones
                        indicating the connecting atoms. The diagonal is set to zero
    """
    n_atoms = positions.shape[0]

    vor = Voronoi(positions)
    dmat = pdist(positions)

    bond_matrix = np.zeros((n_atoms, n_atoms), dtype=bool)
    ridge_points = vor.ridge_points
    bond_matrix[ridge_points[:,0], ridge_points[:,1]] = True
    # since the voronoi algorithm does not double-count 
    # the connections, they are artificially added here
    bond_matrix[ridge_points[:,1], ridge_points[:,0]] = True

    return bond_matrix

###

def get_scaffold(shape = "ico", i = 3, latticeconstant = 3.0,
    energies = [0.5,0.4,0.3], surfaces = [(1, 0, 0), (1, 1, 1), (1, 1, 0)],
    max_bondlength = None):
    """Builds a scaffold of ghost atoms in icosahedral, octahedral or wulff-shape. 
    Takes a shape argument (string can be ico, octa or wulff) as well as 
    the size argument i (int) and a latticeconstant.
    When shape = 'wulff', it is required to give energies and surfaces as lists of equal length.
    Returns a Cluster object with atom type 'X'.

    Args:
        shape (str) :       nanocluster shape such as "ico" (default), "octa", "wulff"
        i (float) :         size of the nanocluster. Has different implications depending 
                            on the shape
        latticeconstant (float) :   lattice constant of the fcc crystal structure defining
                                    the scaling of the nanocluster 
        energies (list) :       Defines Wulff-shape, ignored otherwise. The proportions of 
                                the surface energies defining the prominence of certain
                                slabs defined by surface (energies and corresponding 
                                surfaces must be in the same order)
        surfaces (list) :       Defines Wulff-shape, ignored otherwise. The Miller-indices
                                of the surface slabs. Their energies define the prominence 
                                of certain slabs (energies and corresponding surfaces must 
                                be in the same order)
        max_bondlength (float) :    distance up to which two atoms are considered
                                    bound. If None, the minimum distance is used
                                    as a guess with a tolerance factor of 0.1.
                                    Works well with equidistant atoms.

    Returns:
        cluskit.Scaffold :  Scaffold object, an enhanced ase.Atoms object
                            with additional attributes such as bond_matrix
    """
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
    return Scaffold(atoms, max_bondlength = max_bondlength)



class Clusterer:
    """ A class that generates binary nanoclusters given a bond matrix and atom positions. 
    It configures the atom types in the cluster using a Monte-Carlo algorithm and
    pseudo-energies (taking into account core-shell segregation effect and different interaction
    strenghts between atoms of different types.

    Args:

        bond_matrix (2d ndarray) :  boolean bond matrix / connectivity matrix with ones
                                    indicating the connecting atoms. The diagonal is set to zero        positions (2D ndarray) :    atomic positions of the structure in Angstrom
        ntypeB (int) :  number of atoms of type B in cluster. This argument controls the composition.
        eAA (float) :   pseudo-energy of A-A interaction
        eAB (float) :   pseudo-energy of A-B interaction
        eBB (float) :   pseudo-energy of B-B interaction
        com (1D ndarray) :  center of mass of the nanocluster. If None,
                            the center of atomic positions is chosen
        coreEnergies (list) :   - eEA (float): pseudo-energy of segregation of A into the core,
                                - eEB (float): pseudo-energy of segregation of B into the core.
    
    Returns:
         cluskit.Clusterer :    cluster generator object. Can be activated
                                by the Evolve method
    """
    def __init__(self, bond_matrix, positions, ntypeB, eAA, eAB, eBB, com=None, coreEnergies=[0,0]):

        self.bondmat = bond_matrix
        self.positions = positions
        self.ntypeB = ntypeB

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
        """ Monte-Carlo iterates nsteps with a temperature of kT.
        The attribute self.atomTypes is changed subsequently

        Args:
            kT (float) :    pseudo-temperature of Monte-Carlo in
                            np.exp(-(energyAfter-energyBefore)/kT)
            nsteps (int) :  number of iterations

        Returns:
            Nonetype : None
        """
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
        """Reset to initial state.
        self.atomTypes is reset to a random distribution
        """
        self.atomTypes = np.zeros(self.positions.shape[0], dtype=np.int32)
        self.atomTypes[0:self.ntypeB] = 1
        np.random.shuffle(self.atomTypes)
    # --- end of Reset --- #

class Scaffold(ase.Atoms):
    """
    A child class of the ase.Atoms class. 
    It is a nanocluster class that stores information of a nanocluster on atom positions 
    as well as bonding, but not the actual atomic types.
    Sets up a descriptor based on the atomic types which have been
    given. Can be changed later by overwriting the self.descriptor_setup
    attribute.

    Args:
        max_bondlength (float) :    distance up to which two atoms are considered
                                    bound. If None, the voronoi tesselation is
                                    used

    Returns:
        cluskit.Scaffold :  Scaffold object, an enhanced ase.Atoms object
                            with additional attributes such as bond_matrix    
    """
    def __init__(self, symbols=None,
                positions=None, numbers=None,
                tags=None, momenta=None, masses=None,
                magmoms=None, charges=None,
                scaled_positions=None,
                cell=None, pbc=None, celldisp=None,
                constraint=None,
                calculator=None,
                info=None,
                max_bondlength=None):

        # TODO: solve inheritance elegantly
        self.ase_object = super(Scaffold, self).__init__(symbols=symbols,
                    positions=positions, numbers=numbers,
                    tags=tags, momenta=momenta, masses=masses,
                    magmoms=magmoms, charges=charges,
                    scaled_positions=scaled_positions,
                    cell=cell, pbc=pbc, celldisp=celldisp,
                    constraint=constraint,
                    calculator=calculator,
                    info=info)
        self.ase_object = ase.Atoms(symbols=symbols,
                    positions=positions, numbers=numbers,
                    tags=tags, momenta=momenta, masses=masses,
                    magmoms=magmoms, charges=charges,
                    scaled_positions=scaled_positions,
                    cell=cell, pbc=pbc, celldisp=celldisp,
                    constraint=constraint,
                    calculator=calculator,
                    info=info)

        if max_bondlength == None:
            self.bond_matrix = _get_voronoi_connectivity(self.ase_object.get_positions())
        else:
            self.bond_matrix = _get_connectivity(self.ase_object.get_positions(), 
                max_bondlength = max_bondlength)

        # setting default values for atomic types
        atomic_numbers = sorted(list(set(self.ase_object.get_atomic_numbers())))

        self.default_A = atomic_numbers[0] 
        if len(atomic_numbers) > 1:
            self.default_B = atomic_numbers[1]
        else:
            self.default_B = 0

        self.default_n_A = np.sum(self.ase_object.get_atomic_numbers() == self.default_A)
        self.default_n_B = len(self.ase_object.get_atomic_numbers()) - self.default_n_A 

        self.com = self.ase_object.get_center_of_mass()
        self.distances_to_com = _get_distances_to_com(self.ase_object)

        if 0 in atomic_numbers:
            # dscribe does not allow 0 as atomic index
            atomic_numbers = np.array(atomic_numbers) + 1

        self.descriptor_setup = dscribe.descriptors.SOAP(
            species=atomic_numbers,
            periodic=False,
            rcut=5.0,
            nmax=8,
            lmax=6,
            sparse=False,
            average=True
            )

        self.evolve_temperature = 0.2
        self.evolve_n_steps = 1000

        return

    def get_unique_clusters(self, eAA,eAB,eBB,cEA,cEB, typeA = None, typeB = None, ntypeB = None, n_clus = 1):
        """Gets n_clus clusters all constructed with the given parameters. It uses the Clusterer
        class to generate the clusters. First, 5 times as many nanoclusters are created,
        then they are reduced to n_clus, keeping the most dissimilar structures.

        The Scaffold.descriptor_setup attribute is used for the similarity metric,
        the Scaffold.bond_matrix attribute is used for the connectivity, and hence 
        interactions (eAA, eAB, eBB) to acquire a configuration suitable to the pseudo-energies

        Args:
            eAA (float): pseudo-energy of A-A interaction
            eAB (float): pseudo-energy of A-B interaction
            eBB (float): pseudo-energy of B-B interaction
            eEA (float): pseudo-energy of segregation of A into the core.
            eEB (float): pseudo-energy of segregation of A into the core.
            typeA (int): element of type A in atomic number of PSE.
            typeB (int): element of type B in atomic number of PSE.
            ntypeB (int): number of atoms of type B in cluster. This argument controls the composition.
            n_clus (int): number of cluster to be returned.

        Returns:
            list :  Most dissimilar clusters (cluskit.Cluster objects) at the given Pseudo-energy
                    parameters.         
        """

        # get default values where needed.

        if not typeA:
            typeA = self.default_A
        if  not typeB:
            typeB = self.default_B
        if  not ntypeB:
            ntypeB = self.default_n_B


        atoms = self.ase_object
        bond_matrix = self.bond_matrix
        desc = self.descriptor_setup
        # making sure atomic numbers are adapted by descriptor
        desc.atomic_numbers = [typeA, typeB]

        final_atoms_list = []
        
        positions = atoms.get_positions()
        coreEnergies = [ cEA, cEB ]
        
        atoms_list = []
        for i in range(0,n_clus*5):
            cluster = Clusterer(bond_matrix, positions, ntypeB, eAA, eAB, eBB, com=None, coreEnergies=coreEnergies)
        
            kT = self.evolve_temperature
            nsteps =  self.evolve_n_steps

            cluster.Evolve(kT, nsteps)
            actual_types = cluster.atomTypes.copy()
            actual_types[actual_types == 0] = typeA
            actual_types[actual_types == 1] = typeB
        
            atoms.set_atomic_numbers(actual_types)
            new_atoms = ase.Atoms(numbers=actual_types, positions=positions)
            new_atoms.info = {"eAA" : eAA, "eAB" : eAB, "eBB" : eBB, "cEA" : cEA, "cEB" : cEB}
            atoms_list.append(new_atoms)


        x = desc.create(atoms_list, n_jobs = 1,  positions=None, verbose=False)

        ranks = cluskit.cluster._rank_fps(x, K = None, greedy =False)
        for i in range(0,n_clus):
            cluskit_atoms = Cluster(atoms_list[ranks[i]])
            final_atoms_list.append(cluskit_atoms)

        cluster.Reset()
        return final_atoms_list

    def get_segregated(self, typeA, typeB, ntypeB, n_clus = 1):
        """Gets n_clus segregated nanoclusters

        Args:
            typeA (int): element of type A in atomic number of PSE.
            typeB (int): element of type B in atomic number of PSE.
            ntypeB (int): number of atoms of type B in cluster. This argument controls the composition.
            n_clus (int): number of cluster to be returned.
        
        Returns:
            list :  Most dissimilar clusters (cluskit.Cluster objects) at the given Pseudo-energy
                    parameters.  
        """
        return self.get_unique_clusters(-1,1,-1,0,0,typeA = typeA , typeB = typeB, ntypeB = ntypeB, n_clus = n_clus)

    def get_core_shell(self, typeA, typeB, ntypeB, n_clus = 1):
        """Gets n_clus core-shell nanoclusters with typeA in the
        core.


        Args:
            typeA (int): element of type A in atomic number of PSE.
            typeB (int): element of type B in atomic number of PSE.
            ntypeB (int): number of atoms of type B in cluster. This argument controls the composition.
            n_clus (int): number of cluster to be returned.
        
        Returns:
            list :  Most dissimilar clusters (cluskit.Cluster objects) at the given Pseudo-energy
                    parameters.  
        """
        return self.get_unique_clusters(0,0,0,1,0,typeA = typeA , typeB = typeB, ntypeB = ntypeB, n_clus = n_clus)

    def get_random(self, typeA, typeB, ntypeB, n_clus = 1):
        """Gets n_clus randomly configured nanoclusters


        Args:
            typeA (int): element of type A in atomic number of PSE.
            typeB (int): element of type B in atomic number of PSE.
            ntypeB (int): number of atoms of type B in cluster. This argument controls the composition.
            n_clus (int): number of cluster to be returned.
        
        Returns:
            list :  Most dissimilar clusters (cluskit.Cluster objects) at the given Pseudo-energy
                    parameters.  
        """
        return self.get_unique_clusters(0,0,0,0,0,typeA = typeA , typeB = typeB, ntypeB = ntypeB, n_clus = n_clus)

    def get_ordered(self, typeA, typeB, ntypeB, n_clus = 1):
        """Gets n_clus 'ordered' nanoclusters. Ordered not in the sense
        of crystals. Ordered crystals can be constructed by slicing bulk
        material.
        This implementation of ordered means near-ordered such that the
        number of interactions A-B are maximized

        Args:
            typeA (int): element of type A in atomic number of PSE.
            typeB (int): element of type B in atomic number of PSE.
            ntypeB (int): number of atoms of type B in cluster. This argument controls the composition.
            n_clus (int): number of cluster to be returned.
        
        Returns:
            list :  Most dissimilar clusters (cluskit.Cluster objects) at the given Pseudo-energy
                    parameters.  
        """
        return self.get_unique_clusters(1,-1,1,0,0, typeA = typeA , typeB = typeB, ntypeB = ntypeB, n_clus = n_clus)

    # single-atom alloy
    # TODO SAA method

    # range of eAA,eAB,eBB,cEA,cEB 
    def get_unique_clusters_in_range(self,
        eAA = [-1,1], eAB = [-1,1], eBB = [-1,1], cEA = [-1,1], cEB = [-1,1],
        typeA = None, typeB = None, ntypeB = None, n_clus = 1):
        """Similar method to get_unique_clusters with an additional loop.
        A parameter grid is generated on which nanoclusters are configured
        at each grid point. The most dissimilar structures are chosen based
        on a similarity metric (given by the self.descriptor_setup attribute).

        The grid is chosen as small as possible. In order to get a finer grid, increase
        n_clus. The nanoclusters are returned in an ordered list such that the most
        dissimilar clusters come first.

        Args:
            eAA (list of 2 floats): pseudo-energy of A-A interaction
            eAB (list of 2 floats): pseudo-energy of A-B interaction
            eBB (list of 2 floats): pseudo-energy of B-B interaction
            eEA (list of 2 floats): pseudo-energy of segregation of A into the core.
            eEB (list of 2 floats): pseudo-energy of segregation of A into the core.
            typeA (int): element of type A in atomic number of PSE.
            typeB (int): element of type B in atomic number of PSE.
            ntypeB (int): number of atoms of type B in cluster. This argument controls the composition.
            n_clus (int):   number of cluster to be returned. Affects the internal coarseness of the
                            parameter grid

        Returns:
            list :  Most dissimilar clusters (cluskit.Cluster objects) in the given Pseudo-energy
                    range. 
        """

        # get default values where needed.

        if not typeA:
            typeA = self.default_A
        if  not typeB:
            typeB = self.default_B
        if  not ntypeB:
            ntypeB = self.default_n_B


        atoms = self.ase_object
        bond_matrix = self.bond_matrix
        desc = self.descriptor_setup
        # making sure atomic numbers are adapted by descriptor
        desc.atomic_numbers = [typeA, typeB]

        final_atoms_list = []
        atoms_list = []

        positions = atoms.get_positions()
        

        # discretizing pseudo-energy search space
        steps = [2,2,2,2,2]
        ranges = np.array([
            eAA[1] - eAA[0],
            eAB[1] - eAB[0],
            eBB[1] - eBB[0],
            cEA[1] - cEA[0],
            cEB[1] - cEB[0],
            ], dtype='float')

        step_sizes = ranges.copy()

        for i in range(100):
            # internal numpy use of complex number, see np.mgrid 
            grid = np.mgrid[
                eAA[0]:eAA[1]:complex(0,steps[0]),
                eAB[0]:eAB[1]:complex(0,steps[0]),
                eBB[0]:eBB[1]:complex(0,steps[0]),
                cEA[0]:cEA[1]:complex(0,steps[0]),
                cEB[0]:cEB[1]:complex(0,steps[0]),
            ]

            # check size
            size = grid[0].ravel().shape[0]

            if size < n_clus:
                idx = np.argmax(step_sizes)
                steps[idx] +=1
                step_sizes[idx] = (ranges[idx] - 1.0) / steps[idx]
            else:
                break

        # looping over different pseudo-energies

        grid_1, grid_2, grid_3, grid_4, grid_5 = grid
        grid_points = np.vstack([grid_1.ravel(), grid_2.ravel(),
            grid_3.ravel(), grid_4.ravel(), grid_5.ravel()]).transpose()

        # 5 floats per grid_point: pseudo-energies eAA, eAB, eBB, cEA and cEB
        #print('shape grid points', grid_points.shape)
        for count, grid_point in enumerate(grid_points):
            coreEnergies = [ grid_point[3], grid_point[4] ]
        
            cluster = Clusterer(bond_matrix, positions, ntypeB, grid_point[0], grid_point[1], grid_point[2], com=None, coreEnergies=coreEnergies)
        
            kT = self.evolve_temperature
            nsteps =  self.evolve_n_steps

            cluster.Evolve(kT, nsteps)
            actual_types = cluster.atomTypes.copy()
            actual_types[actual_types == 0] = typeA
            actual_types[actual_types == 1] = typeB
        
            atoms.set_atomic_numbers(actual_types)
            new_atoms = ase.Atoms(numbers=actual_types, positions=positions)
            new_atoms.info = {"eAA" : grid_point[0], "eAB" : grid_point[1], "eBB" : grid_point[2], "cEA" : grid_point[3], "cEB" : grid_point[4]}
            atoms_list.append(new_atoms)


        x = desc.create(atoms_list, n_jobs = 1,  positions=None, verbose=False)

        ranks = cluskit.cluster._rank_fps(x, K = None, greedy =False)
        for i in range(0,n_clus):
            cluskit_atoms = Cluster(atoms_list[ranks[i]])
            final_atoms_list.append(cluskit_atoms)

        cluster.Reset()
        return final_atoms_list
