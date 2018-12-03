from __future__ import absolute_import, division, print_function, unicode_literals

try:
	from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)
except ImportError:
    from __builtin__ import (bytes, str, open, super, range, zip, round, input, int, pow, object)


import sys
import unittest

# Import the test modules
import generaltests


# Initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# Add tests to the test suite
suite.addTests(loader.loadTestsFromModule(generaltests))

# Initialize a runner, pass it the suite and run it
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)

# We need to return a non-zero exit code for the CI to detect errors
sys.exit(not result.wasSuccessful())