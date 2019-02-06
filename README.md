# ClusKit (former ClusGeo)

[![Build Status](https://travis-ci.org/SINGROUP/cluskit.svg?branch=master)](https://travis-ci.org/SINGROUP/cluskit)
[![Coverage Status](https://coveralls.io/repos/github/SINGROUP/cluskit/badge.svg?branch=master)](https://coveralls.io/github/SINGROUP/cluskit?branch=master)


ClusKit is a python package dedicated to surface science on nanoclusters. ClusKit currently focuses on the population of given nanoclusters with adsorbates. Furthermore, it also features the generation of a multitude of different types of nanoclusters.

Sites on the nanoclusters are classified as top, bridge and hollow. One can automatically detect surface atoms as well as those sites on arbitrary nanoclusters. Using a structural descriptor those surface atoms or sites can be compared with each other with respect to their (dis)similarity.

ClusKit is especially designed for creating many samples of adsorbates on a variety of nanoclusters without the need of visually checking the structures. It is complementary to the python package for surface slabs CatKit [https://github.com/SUNCAT-Center/CatKit](https://github.com/SUNCAT-Center/CatKit)


There is no comprehensive documentation yet at this stage, however, the example in the folder examples covers many features. 

Here is a list of the packages current capabilities:
- Generate a scaffold structure (atomic types undetermined) of Icosahedral, Octahedral or Wulff-shape.
- Generate a family of binary clusters (AB) for given pseudo-energy parameters concerning interactions between A and B as well as core-shell segregation.
- Generate a family of the most dissimilar clusters for a range of pseudo-energies
- Get surface atom indices
- Get nonsurface atom indices
- Get all possible sites of class top,bridge or hollow.

- Eliminate a single atom adsorbate as soon as it is too close to another.=

- Get a matrix of size NxM where N is the number of atoms in the cluster and M is the number of descriptor features

- Given a cluster and position of adsorbates, get a matrix of size NxM where N is the number of adsorbates on the cluster and M is the number of descriptor features

- Given a certain threshold, eliminate too similar sites in descriptor feature space.
- Similar to unique sites, but instead of giving a threshold, all sites can be ranked or optionally only the top K can be determined. Ranking is done using farthest point sampling.


# Example

An extensive example is available at examples/example_different_methods.py

Here is a quick example to get you started!
```python
import numpy as np
import ase
from ase.build import molecule
from ase.cluster.icosahedron import Icosahedron
import cluskit

### Make a cluskit object from ase ###
atoms = Icosahedron('Cu', noshells=3)
copper_cluster = cluskit.cluster(atoms)

### Two ways to make a scaffold ###
scaffold_from_ase = cluskit.build.Scaffold(atoms)

scaffold = cluskit.build.get_scaffold(shape = "ico", i = 3, latticeconstant = 3.0,
    energies = [0.5,0.4,0.3], surfaces = [(1, 0, 0), (1, 1, 1), (1, 1, 0)])


### Build binary clusters ###
cluster_list = scaffold.get_segregated(typeA = 28, typeB = 78, ntypeB = 13, n_clus = 2)
print(len(cluster_list))

### Get unique top adsorption sites ###
cluster = cluster_list[0]
sitepositions = cluster.get_sites(1)
sitedescmatrix = cluster.get_sites_descriptor(sitetype = 1)

unique_lst = cluster.get_unique_sites(sitetype = 1, threshold = 0.01)

unique_pos = sitepositions[unique_lst]
adsorbates = ase.Atoms('H' * len(unique_lst), unique_pos)
h_structure = atoms + adsorbates

print(unique_lst.shape)

ase.visualize.view(h_structure)
```

# Dependencies and installation

All dependencies are install automatically. ClusKit depends heavily on ASE, numpy, scipy and to a smaller degree on dscribe. The latter dependency is a descriptor package which you should be familiar with when you use descriptors other than the default method of ClusKit. DScribe is a python package for creating machine learning descriptors for atomistic systems. For more details and tutorials, visit the homepage at:
[https://singroup.github.io/dscribe/](https://singroup.github.io/dscribe/)



The newest version is available from github.
Clone the repository. Inside it run 
```sh
python3 setup.py install
```

A stable version is available through 
```sh
pip install cluskit
```


# Tests

ClusKit now includes regtests and code coverage. 

They range from cluster generation, cluster similarity, spotting surface atoms, finding top, bridge and hollow sites, ranking sites to determining unique sites.


# Contributors

Eiaki Morooka
Marc Jäger
Yashasvi Ranawat
