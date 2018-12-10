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
atoms = Icosahedron('Au', noshells=3)

class SurfaceAtomsTests(unittest.TestCase):

    def test_n_surface_atoms(self):
        """Tests that an ASE Atoms is succesfully converted to a NanoCluster object.
        """
        from clusgeo import ClusGeo
        cluster = ClusGeo(atoms)
        surface_atoms = cluster.get_surface_atoms(bubblesize = 2.5)

        n_surface_atoms = len(surface_atoms)

        self.assertTrue(n_surface_atoms == 42)


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SurfaceAtomsTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)

    # We need to return a non-zero exit code for the gitlab CI to detect errors
    sys.exit(not result.wasSuccessful())
