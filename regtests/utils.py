from __future__ import absolute_import, division, print_function, unicode_literals
try:
    from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)
except ImportError:
    from __builtin__ import (bytes, str, open, super, range, zip, round, input, int, pow, object)

import sys
import numpy as np
import unittest
from ase.build import molecule
import ase
from ase.visualize import view
from ase.build import molecule
from ase.cluster.icosahedron import Icosahedron
import cluskit

atoms = Icosahedron('Cu', noshells=3)

class DudensteinSurfaceTests(unittest.TestCase):
    """Tests the Dudenstein algorithm to find the cluster surface.
    The algorithm bombards the clusters with rays from all sides.
    Includes tests on utility functions using the Dudenstein algorithm.
    """
    def test_get_surface_atoms(self):
        """
        Tests the Dudenstein algorithm to find the cluster surface.
        """
        from cluskit.utils import get_surface_atoms
        surfatoms = get_surface_atoms(atoms)

        self.assertTrue(len(surfatoms), 42)

    def test_get_nonsurface_atoms(self):
        """
        Tests the Dudenstein algorithm to find the 
        cluster nonsurface atoms.
        """
        from cluskit.utils import get_nonsurface_atoms
        nonsurfatoms = get_nonsurface_atoms(atoms)

        self.assertTrue(len(nonsurfatoms), 13)

    def test__get_top_sites(self):
        """Tests finding top sites using the Dudenstein algorithm.
        """
        from cluskit.utils import get_surface_atoms
        surfatoms = get_surface_atoms(atoms)

        from cluskit.utils import _get_top_sites

        sites = _get_top_sites(atoms, surfatoms)

        return

    def test__get_bridge_sites(self):
        from cluskit.utils import get_surface_atoms
        surfatoms = get_surface_atoms(atoms)

        from cluskit.utils import _get_bridge_sites

        sites = _get_bridge_sites(atoms, surfatoms)

        return

    def test__get_hollow_sites(self):
        from cluskit.utils import get_surface_atoms
        surfatoms = get_surface_atoms(atoms)

        from cluskit.utils import _get_hollow_sites

        sites = _get_hollow_sites(atoms, surfatoms)

        return



class MoleculeUtilsTests(unittest.TestCase):

    def test_place_molecule_on_site(self):
        """Tests the function place_molecule_on_site"""
        atoms = Icosahedron('Cu', noshells=3)
        cluster = cluskit.Cluster(atoms)

        zero_site = cluster.get_positions()[53]
        arbitrary_vector = [-2,-2,-2]
        adsorbate_x = ase.Atoms('HHCX', positions=[[2,0,0], [0,2,0], [0,0,0], [-1.4,-1.4, 0]])
        
        adsorbate = cluskit.utils.place_molecule_on_site(molecule = adsorbate_x, 
            zero_site = zero_site, adsorption_vector = arbitrary_vector)

        clus_ads = atoms + adsorbate
        adsorbate_vector_ase = ase.Atoms('OO', positions= [zero_site + arbitrary_vector, 
            zero_site + np.multiply(arbitrary_vector, 2)])
        return

    def test_place_and_preoptimize_adsorbates(self):
        """Tests the function place_and_preoptimize_adsorbates"""
        ### Make a cluskit object from ase ###
        import random
        cluster_atoms = Icosahedron('Cu', noshells=3)
        cluster = cluskit.Cluster(cluster_atoms)
        sitepositions = cluster.get_sites(1)
        sitedescmatrix = cluster.get_sites_descriptor(sitetype = 1)

        ## create molecule with anchor ##
        test_molecules = ['butadiene', 'CH3CH2OCH3', 'C2H6CHOH', 
            'C6H6', 'isobutane', 'C3H8', 'C2H6', 'trans-butane']
        # select only one molecule because of time-intensive function
        adsorbate_x = molecule(random.choice(test_molecules))
        
        ch_sym = adsorbate_x.get_chemical_symbols()
        ch_sym[-1] = 'X'
        adsorbate_x.set_chemical_symbols(ch_sym)

        adsorbates = cluskit.utils.place_and_preoptimize_adsorbates(cluster, 
            adsorbate_x, 1, max_distance=1.5, n_remaining=40, 
            is_reduce=False, is_reset=True, n_lj_steps=2)
        
        #combined_adsorbates = adsorbates[0]
        #for ads in adsorbates[1:]:
        #    combined_adsorbates += ads

        #atoms = combined_adsorbates
        #print(atoms)
        #view(adsorbate_x)
        #view(cluster_atoms + atoms)


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(DudensteinSurfaceTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(MoleculeUtilsTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)

    # We need to return a non-zero exit code for the gitlab CI to detect errors
    sys.exit(not result.wasSuccessful())
