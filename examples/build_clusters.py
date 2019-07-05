from cluskit.build import Scaffold, get_scaffold
from ase.cluster.icosahedron import Icosahedron
from ase.visualize import view
import dscribe

# Parameters: get_unique_clusters
""" 
Args:
    eAA (float): pseudo-energy of A-A interaction
    eAB (float): pseudo-energy of A-B interaction
    eBB (float): pseudo-energy of B-B interaction
    eEA (float): pseudo-energy of segregation of A into the core.
    eEB (float): pseudo-energy of segregation of A into the core.
    typeA (int): element of type A in atomic number of PSE.
    typeB (int): element of type B in atomic number of PSE.
    ntype (int): number of atoms of type B in cluster. This argument controls the composition.
    n_clus (int): number of cluster to be returned.

Returns:
    list of Cluster objects
"""
# Parameters: get_unique_clusters_in_range
"""
Args:
    eAA (list of 2 floats): pseudo-energy of A-A interaction
    eAB (list of 2 floats): pseudo-energy of A-B interaction
    eBB (list of 2 floats): pseudo-energy of B-B interaction
    eEA (list of 2 floats): pseudo-energy of segregation of A into the core.
    eEB (list of 2 floats): pseudo-energy of segregation of A into the core.
    typeA (int): element of type A in atomic number of PSE.
    typeB (int): element of type B in atomic number of PSE.
    ntype (int): number of atoms of type B in cluster. This argument controls the composition.
    n_clus (int): number of cluster to be returned.

Returns:
    list (Cluster): Most dissimilar clusters in the given Pseudo-energy
    range. 
"""

# two ways to create a scaffold (required for cluster generator) 
atoms = Icosahedron('Cu', noshells=3) # or other ase object e.g. structure from file
scaffold_from_ase = Scaffold(atoms)

scaffold = get_scaffold(shape = "ico", i = 3, latticeconstant = 3.0,
    energies = [0.5,0.4,0.3], 
    surfaces = [(1, 0, 0), (1, 1, 1), (1, 1, 0)])

# setup descriptor
scaffold_from_ase.descriptor_setup = dscribe.descriptors.SOAP(
    species=[28, 78],
    periodic=False,
    rcut=5.0,
    nmax=8,
    lmax=6,
    sparse=False,
    average=True
)

scaffold.descriptor_setup = dscribe.descriptors.SOAP(
    species=[28, 78],
    periodic=False,
    rcut=5.0,
    nmax=8,
    lmax=6,
    sparse=False,
    average=True
)


# get clusters with certain parameters
atoms_list = scaffold_from_ase.get_unique_clusters(0,0,0,1,0, 
    typeA = 28, typeB = 78, ntypeB = 13, n_clus = 2)
atoms_list = scaffold.get_unique_clusters(0,0,0,1,0, 
    typeA = 28, typeB = 78, ntypeB = 13, n_clus = 2)

# get clusters in parameter range
atoms_list = scaffold.get_unique_clusters_in_range(typeA = 28, 
    typeB = 78, ntypeB = 13, n_clus = 5)

# get specific distribution types of clusters
atoms_list = scaffold.get_segregated(typeA = 28, 
    typeB = 78, ntypeB = 13, n_clus = 2)
atoms_list = scaffold.get_core_shell(typeA = 28, 
    typeB = 78, ntypeB = 13, n_clus = 2)
atoms_list = scaffold.get_random(typeA = 28, 
    typeB = 78, ntypeB = 13, n_clus = 2)
atoms_list = scaffold.get_ordered(typeA = 28, 
    typeB = 78, ntypeB = 13, n_clus = 10)

if __name__ == '__main__':        
    for atoms in atoms_list:
        view(atoms)
