from __future__ import absolute_import, division, print_function, unicode_literals
try:
    from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)
except ImportError:
    from __builtin__ import (bytes, str, open, super, range, zip, round, input, int, pow, object)

import sys, os
import numpy as np
import unittest

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

class ExamplesTests(unittest.TestCase):

    def test_build_clusters(self):
        """Tests build_clusters.py
        """
        print("test_build_clusters")
        with cd("examples"):
            import examples.build_clusters


    def test_nh3_on_cluster(self):
        """Tests nh3_on_cluster.py
        """
        print("test_nh3_on_cluster")
        with cd("examples"):
            import examples.nh3_on_cluster


    def test_example_different_methods(self):
        """Tests example_different_methods.py
        """
        print("test_example_different_methods")
        with cd("examples"):
            import examples.example_different_methods


    def test_quick_guide(self):
        """Tests quick_guide.py
        """
        print("test_quick_guide")
        with cd("examples"):
            import examples.quick_guide

if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(ExamplesTests))

    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)

    # We need to return a non-zero exit code for the gitlab CI to detect errors
    sys.exit(not result.wasSuccessful())
