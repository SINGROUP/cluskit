from __future__ import absolute_import, division, print_function, unicode_literals
try:
    from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)
except ImportError:
    from __builtin__ import (bytes, str, open, super, range, zip, round, input, int, pow, object)

import sys
import numpy as np
import unittest
from ase.build import molecule

class ASETests(unittest.TestCase):

    def test_import(self):
        """Tests that an ASE Atoms is succesfully converted to a NanoCluster object.
        """
        from cluskit import Cluster

        methanol = molecule('CH3OH')
        nc_methanol = Cluster(methanol.copy()) # not implemented yet

        self.assertTrue(np.array_equal(methanol.get_positions(), nc_methanol.get_positions()))


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(ASETests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)

    # We need to return a non-zero exit code for the gitlab CI to detect errors
    sys.exit(not result.wasSuccessful())
