from __future__ import absolute_import, division, print_function, unicode_literals
try:
    from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)
except ImportError:
    from __builtin__ import (bytes, str, open, super, range, zip, round, input, int, pow, object)

import sys
import numpy as np
import unittest
from ase.build import molecule

from ase.cluster.icosahedron import Icosahedron
atoms = Icosahedron('Cu', noshells=3)

from cluskit import Cluster
import cluskit
import dscribe

cluster = Cluster(atoms)

scaffold = cluskit.build.get_scaffold(shape = "ico", i = 3, latticeconstant = 3.0,
    energies = [0.5,0.4,0.3], surfaces = [(1, 0, 0), (1, 1, 1), (1, 1, 0)])

# descriptor needs to be set correctly
scaffold.descriptor_setup = dscribe.descriptors.SOAP(
    species=[28,78],
    periodic=False,
    rcut=5.0,
    nmax=8,
    lmax=6,
    sparse=False,
    average=True
    )
class ScaffoldTests(unittest.TestCase):

    def test_get_scaffold(self):
        """Tests the .get_scaffold() function.
        """

        scaffold = cluskit.build.get_scaffold(shape = "ico", i = 3, latticeconstant = 3.0,
        energies = [0.5,0.4,0.3], surfaces = [(1, 0, 0), (1, 1, 1), (1, 1, 0)])

        self.assertTrue(len(scaffold) == 55)

        scaffold_from_ase = cluskit.build.Scaffold(atoms)

        self.assertTrue(len(scaffold_from_ase) == 55)


    def test_connectivities(self):
        """Tests both voronoi and maximum bond length connectivity by 
        comparing them to each other.
        """
        atoms = Icosahedron('Cu', noshells=2)

        scaffold_from_ase1 = cluskit.build.Scaffold(atoms, max_bondlength = 2.9)
        scaffold_from_ase2 = cluskit.build.Scaffold(atoms, max_bondlength = None)
        
        #print("#################")
        #print(scaffold_from_ase1.bond_matrix)
        #print("#################")
        #print(scaffold_from_ase2.bond_matrix)
        #print("#################")
        #print(np.sum(scaffold_from_ase1.bond_matrix, axis = 0), np.sum(scaffold_from_ase1.bond_matrix, axis = 1), np.sum(scaffold_from_ase1.bond_matrix))
        #print(np.sum(scaffold_from_ase2.bond_matrix, axis = 0), np.sum(scaffold_from_ase2.bond_matrix, axis = 1), np.sum(scaffold_from_ase2.bond_matrix))

        #print(np.where(scaffold_from_ase1.bond_matrix != scaffold_from_ase2.bond_matrix))
        self.assertTrue( np.all(
            scaffold_from_ase1.bond_matrix == scaffold_from_ase2.bond_matrix)
            )
        
        scaffold1 = cluskit.build.get_scaffold(shape = "ico", i = 3, latticeconstant = 3.0,
            max_bondlength = 2.9)
        scaffold2 = cluskit.build.get_scaffold(shape = "ico", i = 3, latticeconstant = 3.0,
            max_bondlength = None)
        

        self.assertTrue( np.all(
            scaffold1.bond_matrix == scaffold2.bond_matrix)
            )



        return


class FixedPseudoTests(unittest.TestCase):

    def test_get_unique_clusters(self):
        """Tests the .get_unique_clusters() function.
        """

        cluster_list = scaffold.get_unique_clusters(0,0,0,0,0, typeA = 28, typeB = 78, ntypeB = 13, n_clus = 1)

        atomic_numbers = cluster_list[0].get_atomic_numbers()

        n_type_B = np.sum(atomic_numbers == 78)
        self.assertTrue(n_type_B == 13)

        cluster_list = scaffold.get_unique_clusters(0,0,0,0,0, typeA = 28, typeB = 78, ntypeB = 13, n_clus = 2)

        self.assertTrue(len(cluster_list) == 2)

    def test_segregation(self):
        from scipy.spatial.distance import cdist, pdist
        from scipy.spatial.distance import squareform
        
        cluster_list = scaffold.get_segregated(typeA = 28, typeB = 78, ntypeB = 13, n_clus = 1)
        atomic_numbers = cluster_list[0].get_atomic_numbers()
        positions = cluster_list[0].get_positions()

        pos_A = positions[atomic_numbers == 28]
        pos_B = positions[atomic_numbers == 78]

        dmat_A = pdist(pos_A)
        dmat_B = pdist(pos_B)
        dmat_AB = cdist(pos_A, pos_B)

        self.assertTrue(np.mean(dmat_A) < np.mean(dmat_AB))
        self.assertTrue(np.mean(dmat_B) < np.mean(dmat_AB))



    def test_core_shell(self):
        
        cluster_list =scaffold.get_core_shell(typeA = 28, typeB = 78, ntypeB = 13, n_clus = 1)
        distances = cluskit.build._get_distances_to_com(cluster_list[0])
        atomic_numbers = cluster_list[0].get_atomic_numbers()

        distances_A = distances[atomic_numbers == 28]
        distances_B = distances[atomic_numbers == 78]

        self.assertTrue(np.mean(distances_A) > np.mean(distances_B))

    def test_other_methods(self):        

        scaffold.get_ordered(typeA = 28, typeB = 78, ntypeB = 13, n_clus = 1)
        scaffold.get_random(typeA = 28, typeB = 78, ntypeB = 13, n_clus = 1)




class RangedPseudoTests(unittest.TestCase):

    def test_get_unique_clusters_in_range(self):
        """Tests the .get_unique_clusters_in_range() function.
        """
        cluster_list = scaffold.get_unique_clusters_in_range(typeA = 28, typeB = 78, ntypeB = 13, n_clus = 6)
        self.assertTrue(len(cluster_list) == 6)



if __name__ == '__main__':
    # import ase
    # from ase.visualize import view

    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(ScaffoldTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(FixedPseudoTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(RangedPseudoTests))
    

    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)

    # We need to return a non-zero exit code for the gitlab CI to detect errors
    sys.exit(not result.wasSuccessful())
