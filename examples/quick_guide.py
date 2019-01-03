import numpy as np
import ase
from ase.build import molecule
from ase.cluster.icosahedron import Icosahedron
import clusgeo

### Make a clusgeo object from ase ###
atoms = Icosahedron('Cu', noshells=3)
copper_cluster = clusgeo.ClusGeo(atoms)

### Two ways to make a scaffold ###
scaffold_from_ase = clusgeo.build.Scaffold(atoms)

scaffold = clusgeo.build.get_scaffold(shape = "ico", i = 3, latticeconstant = 3.0,
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
