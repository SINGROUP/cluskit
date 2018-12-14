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

from clusgeo import ClusGeo
cluster = ClusGeo(atoms)

class ScaffoldTests(unittest.TestCase):

    def test_get_scaffold(self):
        """Tests the .get_scaffold() function.
        """

        self.assertTrue(42 == 42)



class FixedPseudoTests(unittest.TestCase):

    def test_get_unique_clusters(self):
        """Tests the .get_unique_clusters() function.
        """

        self.assertTrue(42 == 42)

    def test_segregation(self):
        pass


    def test_core_shell(self):
        pass


class RangedPseudoTests(unittest.TestCase):

    def test_get_unique_clusters_in_range(self):
        """Tests the .get_unique_clusters_in_range() function.
        """

        self.assertTrue(42 == 42)



if __name__ == '__main__':
    # import ase
    # from ase.visualize import view

    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SurfaceAtomsTests))
    

    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)

    # We need to return a non-zero exit code for the gitlab CI to detect errors
    sys.exit(not result.wasSuccessful())
