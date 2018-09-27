import os,sys,inspect

#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#sys.path.insert(0,parentdir)

import clusgeo.surface, clusgeo.environment
import ase
import numpy as np

atoms = ase.io.read("au40cu40.xyz")
atoms = ase.io.read("pureSymFe4icos.xyz")

surfaceAtoms = clusgeo.surface.get_surface_atoms(atoms, bubblesize = 2.7)
#nonSurf = clusgeo.surface.get_nonsurf_atoms(atoms, bubblesize = 2.5)
#print("non-surface atoms")
#print(nonSurf)

atnum = atoms.get_atomic_numbers()
atnum[surfaceAtoms] = 103
atoms.set_atomic_numbers(atnum)
ase.io.write("test.xyz", atoms)


print(surfaceAtoms)
atoms = ase.io.read("au40cu40.xyz")
atoms = ase.io.read("Au-icosahedron-3.xyz")
atoms = ase.io.read("pureSymFe4icos.xyz")

soapmatrix = clusgeo.environment.get_soap_cluster(atoms, only_surface=False, bubblesize=2.5, NradBas=10, Lmax =9)
surfsoapmatrix = clusgeo.environment.get_soap_cluster(atoms, only_surface=True, bubblesize=2.5, NradBas=10, Lmax =9)

print("soap cluster shape", soapmatrix.shape)
surfatoms = clusgeo.surface.get_surface_atoms(atoms)
sitepositions = clusgeo.surface.get_top_sites(atoms, surfatoms)
sitesoapmatrix = clusgeo.environment.get_soap_sites(atoms, sitepositions, NradBas=10, Lmax =9)

print("soap sites shape", sitesoapmatrix.shape)

# bridge sites
bridgesite_positions = clusgeo.surface.get_edge_sites(atoms, surfatoms)
adsites = ase.Atoms(['H'] * bridgesite_positions.shape[0], positions = bridgesite_positions)

bridge_cluster = atoms + adsites

ase.io.write("bridge_cluster.xyz", bridge_cluster)

# hollow sites
hollowsite_positions = clusgeo.surface.get_hollow_sites(atoms, surfatoms)
adsites = ase.Atoms(['H'] * hollowsite_positions.shape[0], positions = hollowsite_positions)

hollow_cluster = atoms + adsites

ase.io.write("hollow_cluster.xyz", hollow_cluster)


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
