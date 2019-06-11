import os,sys,inspect
import cluskit
import ase
import numpy as np
from cluskit import Cluster
from ase.visualize import view

### Read structure from xyz into ase ###
atoms = ase.io.read("example_structures/au40cu40.xyz")
#atoms = ase.io.read("example_structures/Au-icosahedron-3.xyz")
#atoms = ase.io.read("example_structures/pureSymFe4icos.xyz")

### Make a Cluster object from ase (same as ase but enhanced) ###
cluster = Cluster(atoms)

### Adsorption site positions ###
sitepositions = cluster.get_sites(1)

# place NH3 on sites
# anchor X defines how molecule binds to cluster 
# (also defines distance to site)
pos = np.array([[ 0.00000000e+00,  0.00000000e+00,  1.16489000e-01],
       [ 0.00000000e+00,  9.39731000e-01, -2.71808000e-01],
       [ 8.13831000e-01, -4.69865000e-01, -2.71808000e-01],
       [-8.13831000e-01, -4.69865000e-01, -2.71808000e-01],
       [ 0.00000000e+00, -1.54520895e-06,  1.91648900e+00]])

adsorbate_x = ase.Atoms('NH3X', positions=pos)
adsorbate_lst = cluster.place_adsorbates(adsorbate_x, 1)

nh3_on_cluster = atoms.copy()
for adsorbate in adsorbate_lst:
    nh3_on_cluster += adsorbate

if __name__ == '__main__':
    view(nh3_on_cluster)


### Unique sites ###
sitedescmatrix = cluster.get_sites_descriptor(sitetype = 1)
print("desc top sites shape", sitedescmatrix.shape)

unique_lst = cluster.get_unique_sites(sitetype = 1, idx=[])
print("unique sites list", unique_lst)
print(unique_lst.shape)


# place NH3 on specific site
idx = unique_lst[0]
top_zero_sites = cluster.zero_site_positions[1]
top_adsorption_vectors = cluster.adsorption_vectors[1]

adsorbate = cluskit.utils.place_molecule_on_site(adsorbate_x, 
    top_zero_sites[idx], top_adsorption_vectors[idx])

if __name__ == '__main__':
    view(atoms + adsorbate)

