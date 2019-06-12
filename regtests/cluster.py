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
cluster = Cluster(atoms)

class SurfaceAtomsTests(unittest.TestCase):

    def test_n_surface_atoms(self):
        """Tests the number of surface atoms.
        """
        surface_atoms = cluster.get_surface_atoms()
        n_surface_atoms = len(surface_atoms)

        self.assertTrue(n_surface_atoms == 42)

    def test_n_nonsurface_atoms(self):
        """Tests the number of core atoms.
        """
        nonsurface_atoms = cluster.get_nonsurface_atoms()
        n_nonsurface_atoms = len(nonsurface_atoms)

        self.assertTrue(n_nonsurface_atoms == 13)

class AdsorptionSitesTests(unittest.TestCase):

    def test_get_sites_return(self):
        """ Tests whether the right number of sites are returned
        using .get_sites() for sitetype = -1,1,2 and 3.
        """
        n_top, n_bridge, n_hollow = 42, 120, 80
        sitepositions = cluster.get_sites(1)
        self.assertTrue(sitepositions.shape[0] == n_top)

        sitepositions = cluster.get_sites(2)
        self.assertTrue(sitepositions.shape[0] == n_bridge)
        sitepositions = cluster.get_sites(3)
        self.assertTrue(sitepositions.shape[0] == n_hollow)
        sitepositions = cluster.get_sites(-1)
        self.assertTrue(sitepositions.shape[0] == n_top + n_bridge + n_hollow)

    def test_get_sites_stored(self):
        """ Tests whether the right number of sites are stored in the Cluster object
        using .get_sites() for sitetype = -1,1,2 and 3.
        """
        n_top, n_bridge, n_hollow = 42, 120, 80
        cluster.get_sites(1)
        self.assertTrue(cluster.site_positions[1].shape[0] == n_top)
        cluster.get_sites(2)
        self.assertTrue(cluster.site_positions[2].shape[0] == n_bridge)
        cluster.get_sites(3)
        self.assertTrue(cluster.site_positions[3].shape[0] == n_hollow)
        cluster.get_sites(-1)
        self.assertTrue(cluster.site_positions[1].shape[0] == n_top)
        self.assertTrue(cluster.site_positions[2].shape[0] == n_bridge)
        self.assertTrue(cluster.site_positions[3].shape[0] == n_hollow)

    def test_get_ase_atomic_adsorbates(self):
        """Tests if a list of adsorbates is returned
        """
        adsorbates = cluster.get_ase_atomic_adsorbates(sitetype = -1, distance = 1.8, atomtype = "H")
        self.assertTrue(type(adsorbates) == list)


    def test_find_closest_site(self):
        """Tests if the closest site to a given point can be found.
        """
        sitepositions1 = cluster.get_sites(1)
        sitepositions2 = cluster.get_sites(2)
        sitepositions3 = cluster.get_sites(3)

        sitetype, idx = cluster.find_closest_site(sitepositions1[0])
        self.assertTrue(sitetype == 1)
        self.assertTrue(idx == 0)

        sitetype, idx = cluster.find_closest_site(sitepositions2[1])
        self.assertTrue(sitetype == 2)
        self.assertTrue(idx == 1)

        sitetype, idx = cluster.find_closest_site(sitepositions3[2])
        self.assertTrue(sitetype == 3)
        self.assertTrue(idx == 2)
        return

    def test_customize_sites(self):
        """Tests the method customize_sites"""
        surface_atoms = cluster.surface_atoms
        sitepositions = cluster.get_sites(-1)

        # take all surface atoms
        custom_sites = cluster.customize_sites(surface_atoms)
        n_top, n_bridge, n_hollow = 42, 120, 80
        self.assertTrue(custom_sites[1].shape[0] == n_top)
        self.assertTrue(custom_sites[2].shape[0] == n_bridge)
        self.assertTrue(custom_sites[3].shape[0] == n_hollow)

        custom_sites = cluster.customize_sites(surface_atoms, sitetype = 1)
        self.assertTrue(custom_sites.shape[0] == n_top)
        custom_sites = cluster.customize_sites(surface_atoms, sitetype = 2)
        self.assertTrue(custom_sites.shape[0] == n_bridge)
        custom_sites = cluster.customize_sites(surface_atoms, sitetype = 3)
        self.assertTrue(custom_sites.shape[0] == n_hollow)

        # check is_exclusive
        custom_sites = cluster.customize_sites(surface_atoms, is_exclusive = True)
        n_top, n_bridge, n_hollow = 42, 120, 80
        self.assertTrue(custom_sites[1].shape[0] == n_top)
        self.assertTrue(custom_sites[2].shape[0] == n_bridge)
        self.assertTrue(custom_sites[3].shape[0] == n_hollow)


        # take only one surface atom
        custom_sites = cluster.customize_sites(surface_atoms[-1])
        self.assertTrue(custom_sites[1].shape[0] == 1)
        # number depends on which type of surface atom 
        # (here is on the edge)
        self.assertTrue(custom_sites[2].shape[0] == 6)
        self.assertTrue(custom_sites[3].shape[0] == 6)


        # take only one surface atom with is_exclusive
        custom_sites = cluster.customize_sites(surface_atoms[-1], is_exclusive = True)
        self.assertTrue(custom_sites[1].shape[0] == 1)
        self.assertTrue(custom_sites[2].shape[0] == 0)
        self.assertTrue(custom_sites[3].shape[0] == 0)
        

        custom_sites = cluster.customize_sites(surface_atoms[:19], is_exclusive = True)


        print(custom_sites)
        return



        
class DefaultDescriptorTests(unittest.TestCase):
    """ Tests if the default descriptor (SOAP) functions properly in
    .get_cluster_descriptor() and .get_sites_descriptor()
    """

    def test_get_cluster_descriptor(self):
        descmatrix = cluster.get_cluster_descriptor(only_surface=False)
        self.assertTrue(cluster.cluster_descriptor.shape[0] == len(atoms))
        self.assertTrue(descmatrix.shape[0] == len(atoms))

        n_top = 42
        descmatrix = cluster.get_cluster_descriptor(only_surface=True)
        self.assertTrue(cluster.cluster_descriptor.shape[0] == len(atoms))
        self.assertTrue(descmatrix.shape[0] == n_top)

    def test_get_sites_descriptor(self):  
        n_top, n_bridge, n_hollow = 42, 120, 80
        sitedescmatrix = cluster.get_sites_descriptor(sitetype = 1)
        self.assertTrue(cluster.sites_descriptor[1].shape[0] == n_top)
        self.assertTrue(sitedescmatrix.shape[0] == n_top)

        sitedescmatrix = cluster.get_sites_descriptor(sitetype = -1)
        self.assertTrue(cluster.sites_descriptor[1].shape[0] == n_top)
        self.assertTrue(cluster.sites_descriptor[2].shape[0] == n_bridge)        
        self.assertTrue(cluster.sites_descriptor[3].shape[0] == n_hollow)        
       
        self.assertTrue(sitedescmatrix.shape[0] == n_top + n_bridge + n_hollow)


class UniquenessTests(unittest.TestCase):
    """ Tests functionality of the 'unique' methods
    .get_unique_cluster_atoms() and .get_unique_sites()
    """

    def test_get_unique_cluster_atoms(self):
        unique_lst = cluster.get_unique_cluster_atoms(threshold = 0.001, idx=[])
        self.assertTrue(len(unique_lst) == 4)

    def test_get_unique_sites(self):    
        unique_lst = cluster.get_unique_sites(sitetype = 1, idx=[])
        self.assertTrue(len(unique_lst) == 2)

        unique_lst = cluster.get_unique_sites(sitetype = 2, threshold = 0.01, idx=[])
        self.assertTrue(len(unique_lst) == 2)

        unique_lst = cluster.get_unique_sites(sitetype = 3, threshold = 0.01, idx=[])
        self.assertTrue(len(unique_lst) == 2)

    def test_idx(self):
        """Tests the use of indices in the function"""
        unique_lst = cluster.get_unique_sites(sitetype = 1, idx=[5])
        self.assertTrue(len(unique_lst) == 1)
        
        unique_lst = cluster.get_unique_sites(sitetype = 1, idx=[5,5,5,5,5,5])
        self.assertTrue(len(unique_lst) == 1)
        
        unique_lst = cluster.get_unique_sites(sitetype = 2, threshold = 0.01, idx=[1,100])
        self.assertTrue(len(unique_lst) == 2)

        unique_lst = cluster.get_unique_sites(sitetype = 3, threshold = 0.01, idx=[0,1,2,3,70,20, 12, 10, 5, 55])
        self.assertTrue(len(unique_lst) == 2)

        
        n_top, n_bridge, n_hollow = 42, 120, 80
        idx = np.arange(n_top + 1)

        with self.assertRaises(IndexError):
            unique_lst = cluster.get_unique_sites(sitetype = 1, idx=idx)





class RankingTests(unittest.TestCase):
    """ Tests functionality of the ranking method
    .get_ranked_sites()
    """

    def test_get_ranked_sites(self):
        # fps ranking testing
        # no double counting
        ranked_ids = cluster.get_ranked_sites(sitetype= -1, K = None, idx=[], greedy = True)
        self.assertTrue(len(ranked_ids) == len(set(ranked_ids)))
        ranked_ids = cluster.get_ranked_sites(sitetype= -1, K = None, idx=[], greedy = False)
        self.assertTrue(len(ranked_ids) == len(set(ranked_ids)))

    def test_idx(self):
        """Tests the use of indices in the function"""
        ranked_ids = cluster.get_ranked_sites(sitetype= -1, K = None, idx=[0, 100, 200], greedy = True)
        self.assertTrue(len(ranked_ids) == len(set(ranked_ids)))
        self.assertTrue(len(ranked_ids) == 3)

        ranked_ids = cluster.get_ranked_sites(sitetype= -1, K = None, idx=[0, 0, 0, 100, 100, 100], greedy = False)
        self.assertTrue(len(ranked_ids) != len(set(ranked_ids)))
        self.assertTrue(len(ranked_ids) == 6)
        self.assertTrue(len(set(ranked_ids)) == 2)

        
        n_top, n_bridge, n_hollow = 42, 120, 80
        idx = np.arange(n_top + 1)

        with self.assertRaises(IndexError):
            unique_lst = cluster.get_ranked_sites(sitetype = 1, idx=idx)




class MoleculeOnSitesTests(unittest.TestCase):

    def test_place_molecules(self):
        """ Tests whether the placement of a simple adsorbate runs without error.
        """
        import ase
        adsorbate_x = ase.Atoms('HHCX', positions=[[2,0,0], [0,2,0], [0,0,0], [-1.4,-1.4, 0]])

        sitepositions = cluster.get_sites(1)

        for i in [-1, 1, 2, 3]:
            adsorbate_lst = cluster.place_adsorbates(adsorbate_x, sitetype = i)
        


if __name__ == '__main__':

    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SurfaceAtomsTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(AdsorptionSitesTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(DefaultDescriptorTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(UniquenessTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(RankingTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(MoleculeOnSitesTests))
    

    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)

    # We need to return a non-zero exit code for the gitlab CI to detect errors
    sys.exit(not result.wasSuccessful())
