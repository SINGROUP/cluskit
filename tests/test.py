import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import clusgeo.surface, clusgeo.environment
import ase
import numpy as np

atoms = ase.io.read("au40cu40.xyz")

surfaceAtoms = clusgeo.surface.get_surface_atoms(atoms, bubblesize = 2.7)


atnum = atoms.get_atomic_numbers()
atnum[surfaceAtoms] = 103
atoms.set_atomic_numbers(atnum)

ase.io.write("test.xyz", atoms)


print(surfaceAtoms)
atoms = ase.io.read("au40cu40.xyz")
atoms = ase.io.read("Au-icosahedron-3.xyz")

soapmatrix = clusgeo.environment.get_soap_cluster(atoms, only_surface=False, bubblesize=2.5, NradBas=10, Lmax =9)
surfsoapmatrix = clusgeo.environment.get_soap_cluster(atoms, only_surface=True, bubblesize=2.5, NradBas=10, Lmax =9)

print("soap cluster shape", soapmatrix.shape)
surfatoms = clusgeo.surface.get_surface_atoms(atoms)
sitepositions = clusgeo.surface.get_top_sites(atoms, surfatoms)
sitesoapmatrix = clusgeo.environment.get_soap_sites(atoms, sitepositions, NradBas=10, Lmax =9)

print("soap sites shape", sitesoapmatrix.shape)

# fps ranking testing

soapmatrix = np.random.rand(5,10)
soapmatrix = np.vstack((soapmatrix,soapmatrix))
print(soapmatrix.shape)

ranked_ids = clusgeo.environment.rank_sites(soapmatrix, K = None, idx=[], greedy = True, is_safe = True)



print(ranked_ids)
print("size of ranked ids:", len(ranked_ids), "set:", len(set(ranked_ids)))
assert len(ranked_ids) == len(set(ranked_ids)), "Error! Double counting in FPS!" 

# unique sites testing

unique_lst = clusgeo.environment.get_unique_sites(sitesoapmatrix, idx=surfatoms)

unique_pos = sitepositions[unique_lst]
adsorbates = ase.Atoms('H' * len(unique_lst), unique_pos)
h_structure = atoms + adsorbates

ase.io.write("uniqueH.xyz", h_structure)

print(unique_lst.shape)
