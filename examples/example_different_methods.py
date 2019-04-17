import os,sys,inspect
import cluskit
import ase
import numpy as np
from cluskit import Cluster

### Read structure from xyz into ase ###
atoms = ase.io.read("example_structures/au40cu40.xyz")
#atoms = ase.io.read("example_structures/Au-icosahedron-3.xyz")
#atoms = ase.io.read("example_structures/pureSymFe4icos.xyz")

### Make a Cluster object from ase (same as ase but enhanced) ###
cluster = Cluster(atoms)

### Get surface and non-surface atoms ###
surface_atoms = cluster.get_surface_atoms()

nonSurf = cluster.get_nonsurface_atoms()
print("surface atoms", len(surface_atoms))
print("non-surface atoms", len(nonSurf))


### Descriptor features for cluster atoms ###
descmatrix = cluster.get_cluster_descriptor(only_surface=False, )
print("get_cluster_descriptor" , descmatrix.shape)
descmatrix = cluster.get_cluster_descriptor(only_surface=True, )
print("get_cluster_descriptor, only surface", descmatrix.shape)

### Adsorption site positions ###
sitepositions = cluster.get_sites(1)
print("top sites", sitepositions.shape)
sitepositions = cluster.get_sites(2)
print("bridge sites", sitepositions.shape)
sitepositions = cluster.get_sites(3)
print("hollow sites", sitepositions.shape)
sitepositions = cluster.get_sites(-1)
print("top, bridge, hollow sites", sitepositions)

sitedescmatrix = cluster.get_sites_descriptor(sitetype = 1)
print("desc top sites shape", sitedescmatrix.shape)

sitedescmatrix = cluster.get_sites_descriptor(sitetype = -1)
print("desc top, bridge, hollow sites shape", sitedescmatrix.shape)


### Fps ranking ###
ranked_ids = cluster.get_ranked_sites(sitetype= -1, K = None, idx=[], greedy = True, )

print(ranked_ids)
print("size of ranked ids:", len(ranked_ids), "set:", len(set(ranked_ids)))
assert len(ranked_ids) == len(set(ranked_ids)), "Error! Double counting in FPS!" 

### Unique sites ###
unique_lst = cluster.get_unique_sites(sitetype = -1, idx=[])

unique_pos = sitepositions[unique_lst]
adsorbates = ase.Atoms('H' * len(unique_lst), unique_pos)
h_structure = atoms + adsorbates

#ase.io.write("uniqueH.xyz", h_structure)
print(unique_lst.shape)


### Unique surface atoms ###
unique_lst = cluster.get_unique_cluster_atoms(idx=surface_atoms)
