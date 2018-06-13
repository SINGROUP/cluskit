import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import clusgeo.surface, clusgeo.environment
import ase

atoms = ase.io.read("au40cu40.xyz")

surfaceAtoms = clusgeo.surface.get_surface_atoms(atoms, bubblesize = 2.7)


atnum = atoms.get_atomic_numbers()
atnum[surfaceAtoms] = 103
atoms.set_atomic_numbers(atnum)

ase.io.write("test.xyz", atoms)


print(surfaceAtoms)
