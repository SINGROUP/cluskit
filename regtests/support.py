from __future__ import absolute_import, division, print_function, unicode_literals
try:
    from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)
except ImportError:
    from __builtin__ import (bytes, str, open, super, range, zip, round, input, int, pow, object)

import sys
import numpy as np
import unittest
from ase.build import molecule

from ase.build import fcc111
slab = fcc111('Al', size=(3,3,3), vacuum=10.0)
atoms = slab
from cluskit import Support
support = Support(slab)
support2 = Support(slab)


class SurfaceAtomsTests(unittest.TestCase):

    def test_n_surface_atoms(self):
        """Tests the number of surface atoms.
        """
        surface_atoms = support.get_surface_atoms()
        n_surface_atoms = len(surface_atoms)

        self.assertTrue(n_surface_atoms == 18)

    def test_n_nonsurface_atoms(self):
        """Tests the number of core atoms.
        """
        nonsurface_atoms = support.get_nonsurface_atoms()
        n_nonsurface_atoms = len(nonsurface_atoms)

        self.assertTrue(n_nonsurface_atoms == 9)

    def test_reduce_surface_atoms(self):
        """Tests the number of surface atoms on one side of the slab.
        """
        surface_atoms = support2.get_surface_atoms()
        n_surface_atoms = len(surface_atoms)

        reduced_surface_atoms = support2.reduce_surface_atoms(z_direction = 1)
        n_reduced = len(reduced_surface_atoms)
        self.assertTrue(n_surface_atoms == 2 * n_reduced)

class AdsorptionSitesTests(unittest.TestCase):

    def test_get_sites_return(self):
        """ Tests whether the right number of sites are returned
        using .get_sites() for sitetype = -1,1,2 and 3.
        """
        surface_atoms = support.get_surface_atoms()

        n_top, n_bridge, n_hollow = 18, 54, 36
        sitepositions = support.get_sites(1)
        self.assertTrue(sitepositions.shape[0] == n_top)

        sitepositions = support.get_sites(2)
        self.assertTrue(sitepositions.shape[0] == n_bridge)
        sitepositions = support.get_sites(3)
        self.assertTrue(sitepositions.shape[0] == n_hollow)
        sitepositions = support.get_sites(-1)
        self.assertTrue(sitepositions.shape[0] == n_top + n_bridge + n_hollow)

    def test_get_sites_stored(self):
        """ Tests whether the right number of sites are stored in the Support object
        using .get_sites() for sitetype = -1,1,2 and 3.
        """
        n_top, n_bridge, n_hollow = 18, 54, 36
        support.get_sites(1)
        self.assertTrue(support.site_positions[1].shape[0] == n_top)
        support.get_sites(2)
        self.assertTrue(support.site_positions[2].shape[0] == n_bridge)
        support.get_sites(3)
        self.assertTrue(support.site_positions[3].shape[0] == n_hollow)
        support.get_sites(-1)
        self.assertTrue(support.site_positions[1].shape[0] == n_top)
        self.assertTrue(support.site_positions[2].shape[0] == n_bridge)
        self.assertTrue(support.site_positions[3].shape[0] == n_hollow)

    def test_get_ase_atomic_adsorbates(self):
        """Tests if a list of adsorbates is returned
        """
        adsorbates = support.get_ase_atomic_adsorbates(sitetype = -1, distance = 1.8, atomtype = "H")
        self.assertTrue(type(adsorbates) == list)


    def test_find_closest_site(self):
        """Tests if the closest site to a given point can be found.
        """
        sitepositions1 = support.get_sites(1)
        sitepositions2 = support.get_sites(2)
        sitepositions3 = support.get_sites(3)

        sitetype, idx = support.find_closest_site(sitepositions1[0])
        self.assertTrue(sitetype == 1)
        self.assertTrue(idx == 0)

        sitetype, idx = support.find_closest_site(sitepositions2[1])
        self.assertTrue(sitetype == 2)
        self.assertTrue(idx == 1)

        sitetype, idx = support.find_closest_site(sitepositions3[2])
        self.assertTrue(sitetype == 3)
        self.assertTrue(idx == 2)
        return

    def test_customize_sites(self):
        """Tests the method customize_sites"""
        surface_atoms = support.surface_atoms
        sitepositions = support.get_sites(-1)

        # take all surface atoms
        custom_sites = support.customize_sites(surface_atoms)
        n_top, n_bridge, n_hollow = 18, 54, 36
        self.assertTrue(custom_sites[1].shape[0] == n_top)
        self.assertTrue(custom_sites[2].shape[0] == n_bridge)
        self.assertTrue(custom_sites[3].shape[0] == n_hollow)

        custom_sites = support.customize_sites(surface_atoms, sitetype = 1)
        self.assertTrue(custom_sites.shape[0] == n_top)
        custom_sites = support.customize_sites(surface_atoms, sitetype = 2)
        self.assertTrue(custom_sites.shape[0] == n_bridge)
        custom_sites = support.customize_sites(surface_atoms, sitetype = 3)
        self.assertTrue(custom_sites.shape[0] == n_hollow)

        # check is_exclusive
        custom_sites = support.customize_sites(surface_atoms, is_exclusive = True)
        n_top, n_bridge, n_hollow = 18, 54, 36
        self.assertTrue(custom_sites[1].shape[0] == n_top)
        self.assertTrue(custom_sites[2].shape[0] == n_bridge)
        self.assertTrue(custom_sites[3].shape[0] == n_hollow)


        # take only one surface atom
        custom_sites = support.customize_sites(surface_atoms[-1])
        self.assertTrue(custom_sites[1].shape[0] == 1)
        # number depends on which type of surface atom 
        # (here is on the edge)
        self.assertTrue(custom_sites[2].shape[0] == 6)
        self.assertTrue(custom_sites[3].shape[0] == 6)


        # take only one surface atom with is_exclusive
        custom_sites = support.customize_sites(surface_atoms[-1], is_exclusive = True)
        self.assertTrue(custom_sites[1].shape[0] == 1)
        self.assertTrue(custom_sites[2].shape[0] == 0)
        self.assertTrue(custom_sites[3].shape[0] == 0)
        
        custom_sites = support.customize_sites(surface_atoms[:19], is_exclusive = True)

        #print(custom_sites)
        return

        
class DefaultDescriptorTests(unittest.TestCase):
    """ Tests if the default descriptor (SOAP) functions properly in
    .get_slab_descriptor() and .get_sites_descriptor()
    """

    def test_get_slab_descriptor(self):
        descmatrix = support.get_slab_descriptor(only_surface=False)
        self.assertTrue(support.slab_descriptor.shape[0] == len(atoms))
        self.assertTrue(descmatrix.shape[0] == len(atoms))

        n_top = 18
        descmatrix = support.get_slab_descriptor(only_surface=True)
        self.assertTrue(support.slab_descriptor.shape[0] == len(atoms))
        self.assertTrue(descmatrix.shape[0] == n_top)

    def test_get_sites_descriptor(self):  
        n_top, n_bridge, n_hollow = 18, 54, 36
        sitedescmatrix = support.get_sites_descriptor(sitetype = 1)
        self.assertTrue(support.sites_descriptor[1].shape[0] == n_top)
        self.assertTrue(sitedescmatrix.shape[0] == n_top)

        sitedescmatrix = support.get_sites_descriptor(sitetype = -1)
        self.assertTrue(support.sites_descriptor[1].shape[0] == n_top)
        self.assertTrue(support.sites_descriptor[2].shape[0] == n_bridge)        
        self.assertTrue(support.sites_descriptor[3].shape[0] == n_hollow)        
       
        self.assertTrue(sitedescmatrix.shape[0] == n_top + n_bridge + n_hollow)


class UniquenessTests(unittest.TestCase):
    """ Tests functionality of the 'unique' methods
    .get_unique_slab_atoms() and .get_unique_sites()
    """

    def test_get_unique_slab_atoms(self):
        unique_lst = support.get_unique_slab_atoms(threshold = 0.001, idx=[])
        self.assertTrue(len(unique_lst) == 2)

    def test_get_unique_sites(self):    
        unique_lst = support.get_unique_sites(sitetype = 1, idx=[])
        self.assertTrue(len(unique_lst) == 1)

        unique_lst = support.get_unique_sites(sitetype = 2, threshold = 0.01, idx=[])
        self.assertTrue(len(unique_lst) == 1)

        unique_lst = support.get_unique_sites(sitetype = 3, threshold = 0.01, idx=[])
        self.assertTrue(len(unique_lst) == 1)

    def test_idx(self):
        """Tests the use of indices in the function"""
        unique_lst = support.get_unique_sites(sitetype = 1, idx=[5])
        self.assertTrue(len(unique_lst) == 1)
        
        unique_lst = support.get_unique_sites(sitetype = 1, idx=[5,5,5,5,5,5])
        self.assertTrue(len(unique_lst) == 1)
        
        unique_lst = support.get_unique_sites(sitetype = 2, threshold = 0.01, idx=[1,30])
        self.assertTrue(len(unique_lst) == 1)

        unique_lst = support.get_unique_sites(sitetype = 3, threshold = 0.01, idx=[0,1,2,3,20, 12, 10, 5,])
        self.assertTrue(len(unique_lst) == 1)

        
        n_top, n_bridge, n_hollow = 18, 54, 36
        idx = np.arange(n_top + 1)

        with self.assertRaises(IndexError):
            unique_lst = support.get_unique_sites(sitetype = 1, idx=idx)



class RankingTests(unittest.TestCase):
    """ Tests functionality of the ranking method
    .get_ranked_sites()
    """

    def test_get_ranked_sites(self):
        # fps ranking testing
        # no double counting
        ranked_ids = support.get_ranked_sites(sitetype= -1, K = None, idx=[], greedy = True)
        self.assertTrue(len(ranked_ids) == len(set(ranked_ids)))
        ranked_ids = support.get_ranked_sites(sitetype= -1, K = None, idx=[], greedy = False)
        self.assertTrue(len(ranked_ids) == len(set(ranked_ids)))

    def test_idx(self):
        """Tests the use of indices in the function"""
        ranked_ids = support.get_ranked_sites(sitetype= -1, K = None, idx=[0, 50, 100], greedy = True)
        self.assertTrue(len(ranked_ids) == len(set(ranked_ids)))
        self.assertTrue(len(ranked_ids) == 3)

        ranked_ids = support.get_ranked_sites(sitetype= -1, K = None, idx=[0, 0, 0, 100, 100, 100], greedy = False)
        self.assertTrue(len(ranked_ids) != len(set(ranked_ids)))
        self.assertTrue(len(ranked_ids) == 6)
        self.assertTrue(len(set(ranked_ids)) == 2)

        n_top, n_bridge, n_hollow = 18, 54, 36
        idx = np.arange(n_top + 1)

        with self.assertRaises(IndexError):
            unique_lst = support.get_ranked_sites(sitetype = 1, idx=idx)




class MoleculeOnSitesTests(unittest.TestCase):

    def test_place_molecules(self):
        """ Tests whether the placement of a simple adsorbate runs without error.
        """
        import ase
        adsorbate_x = ase.Atoms('HHCX', positions=[[2,0,0], [0,2,0], [0,0,0], [-1.4,-1.4, 0]])

        sitepositions = support.get_sites(1)

        for i in [-1, 1, 2, 3]:
            adsorbate_lst = support.place_adsorbates(adsorbate_x, sitetype = i)
        


if __name__ == '__main__':

    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SurfaceAtomsTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(AdsorptionSitesTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(DefaultDescriptorTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(MoleculeOnSitesTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(UniquenessTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(RankingTests))
    
    

    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)

    # We need to return a non-zero exit code for the gitlab CI to detect errors
    sys.exit(not result.wasSuccessful())
