import os,sys,inspect

#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#sys.path.insert(0,parentdir)

import clusgeo
import ase
import numpy as np
from clusgeo import ClusGeo

atoms = ase.io.read("example_structures/au40cu40.xyz")
#atoms = ase.io.read("example_structures/Au-icosahedron-3.xyz")
#atoms = ase.io.read("example_structures/pureSymFe4icos.xyz")

cluster = ClusGeo(atoms)

surface_atoms = cluster.get_surface_atoms(bubblesize = 2.7)

nonSurf = cluster.get_nonsurface_atoms(bubblesize = 2.7)
print("surface atoms", len(surface_atoms))
print("non-surface atoms", len(nonSurf))
#print(nonSurf)

#atnum = atoms.get_atomic_numbers()
#atnum[surface_atoms] = 103
#atoms.set_atomic_numbers(atnum)
#ase.io.write("test.xyz", atoms)

descmatrix = cluster.get_cluster_descriptor(only_surface=False, bubblesize=2.7)
print("get_cluster_descriptor" , descmatrix.shape)
descmatrix = cluster.get_cluster_descriptor(only_surface=True, bubblesize=2.7)
print("get_cluster_descriptor, only surface", descmatrix.shape)

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


# fps ranking testing
ranked_ids = cluster.get_ranked_sites(sitetype= -1, K = None, idx=[], greedy = True, is_safe = True)

print(ranked_ids)
print("size of ranked ids:", len(ranked_ids), "set:", len(set(ranked_ids)))
assert len(ranked_ids) == len(set(ranked_ids)), "Error! Double counting in FPS!" 

# unique sites testing
unique_lst = cluster.get_unique_sites(sitetype = -1, idx=[])

unique_pos = sitepositions[unique_lst]
adsorbates = ase.Atoms('H' * len(unique_lst), unique_pos)
h_structure = atoms + adsorbates

#ase.io.write("uniqueH.xyz", h_structure)

print(unique_lst.shape)


# unique surface sites testing
unique_lst = cluster.get_unique_cluster_atoms(idx=surface_atoms)

unique_pos = sitepositions[unique_lst]
adsorbates = ase.Atoms('H' * len(unique_lst), unique_pos)
h_structure = atoms + adsorbates

#ase.io.write("uniqueH.xyz", h_structure)


print(unique_lst.shape)
