import numpy as np 
import ase, ase.io
import os, argparse, glob
from ctypes import *
import pathlib
from scipy.spatial.distance import cdist

_PATH_TO_CLUSGEO_SO = os.path.dirname(os.path.abspath(__file__))
_CLUSGEO_SOFILES = glob.glob( "".join([ _PATH_TO_CLUSGEO_SO, "/../lib/libclusgeo3.*so*"]) )
_LIBCLUSGEO = CDLL(_CLUSGEO_SOFILES[0])


def _format_ase2clusgeo(obj, all_atomtypes=[]):
    """ Takes an ase Atoms object and returns numpy arrays and integers
    which are read by the internal clusgeo. Apos is currently a flattened
    out numpy array
    """
    #atoms metadata
    totalAN = len(obj)
    if all_atomtypes:
        atomtype_set = set(all_atomtypes)
    else:
        atomtype_set = set(obj.get_atomic_numbers())
    num_atomtypes = len(atomtype_set)

    atomtype_lst = np.sort(list(atomtype_set))
    n_atoms_per_type_lst = []
    pos_lst = []
    for atomtype in atomtype_lst:
        condition = obj.get_atomic_numbers() == atomtype
        pos_onetype = obj.get_positions()[condition]
        n_onetype = pos_onetype.shape[0]

        # store data in lists
        pos_lst.append(pos_onetype)
        n_atoms_per_type_lst.append(n_onetype)

    typeNs = n_atoms_per_type_lst
    Ntypes = len(n_atoms_per_type_lst)
    atomtype_lst
    Apos = np.concatenate(pos_lst).ravel()
    return Apos, typeNs, Ntypes, atomtype_lst, totalAN

class ClusGeo(ase.Atoms):

    def __init__(self, symbols=None,
                positions=None, numbers=None,
                tags=None, momenta=None, masses=None,
                magmoms=None, charges=None,
                scaled_positions=None,
                cell=None, pbc=None, celldisp=None,
                constraint=None,
                calculator=None,
                info=None,
                surface=None):

     super().__init__(symbols=symbols,
                positions=positions, numbers=numbers,
                tags=tags, momenta=momenta, masses=masses,
                magmoms=magmoms, charges=charges,
                scaled_positions=scaled_positions,
                cell=cell, pbc=pbc, celldisp=celldisp,
                constraint=constraint,
                calculator=calculator,
                info=info)

     if surface is not None:
         if isinstance(surface[0], bool) or isinstance(surface[0], np.bool_):
             self.arrays['surface'] = np.array(surface, dtype='bool')
         elif isinstance(surface[0], list):
             surface_hold = np.zeros(len(self.arrays['positions']), dtype='bool')
             surface_hold[surface] = True
             self.arrays['surface'] = surface_hold
     else:
         _ = self.get_surface_atoms()

    def get_surface_atoms(self, bubblesize = 2.5, bool=False):
        """Takes an ASE atoms object and a bubblesize determining how concave a surface can be.
        Returns an array of indices of surface atoms.
        """
        # get clusgeo internal format for c-code
        if self.has('surface'):
            if bool:
                return self.arrays['surface']
            else:
                return np.nonzero(self.arrays['surface'])[0]

        py_totalAN = len(self)
        py_surfAtoms = np.zeros(py_totalAN, dtype=int)
        pos = self.get_positions()
        py_x, py_y, py_z = pos[:,0], pos[:,1], pos[:,2]

        # convert int to c_int
        totalAN = c_int(py_totalAN)
        # convert double to c_double
        bubblesize = c_double(float(bubblesize))
        #convert int array to c_int array
        surfAtoms = (c_int * py_totalAN)(*py_surfAtoms)

        # convert to c_double arrays
        x = (c_double * py_totalAN)(*py_x)
        y = (c_double * py_totalAN)(*py_y)
        z = (c_double * py_totalAN)(*py_z)

        _LIBCLUSGEO.findSurf.argtypes = [POINTER (c_double),POINTER (c_double), POINTER (c_double), POINTER (c_int), c_int, c_double]
        _LIBCLUSGEO.findSurf.restype = c_int

        Nsurf = _LIBCLUSGEO.findSurf(x, y, z, surfAtoms, totalAN, bubblesize)

        py_surfAtoms = np.ctypeslib.as_array( surfAtoms, shape=(py_totalAN))
        py_surfAtoms = py_surfAtoms[:Nsurf]

        surface_hold = np.zeros(len(self), dtype='bool')
        surface_hold[py_surfAtoms] = True
        self.arrays['surface'] = surface_hold

        if bool:
            return surface_hold
        else:
            return py_surfAtoms
    ####### NEW 2018/Jun/26 - Aki########################################################
    def get_nonsurface_atoms(self, bubblesize = 2.5):
        """Takes an ASE atoms object and a bubblesize determining how concave a surface can be.
        Returns an array of indices of surface atoms.
        """
        # get clusgeo internal format for c-code
        surface = self.get_surface_atoms(bool=True)
        return np.nonzero(np.logical_not(surface))[0]

    def _get_top_sites(self, distance=1.5):
        """Takes an ASE atoms object, an array of surface atom indices and a distance as input
        Returns a 2D-array of top site positions with the defined distance from the surface.
        """
        # get clusgeo internal format for c-code
        py_totalAN = len(self)
        surfatoms = self.get_surface_atoms()
        py_Nsurf = len(surfatoms)
        py_surfAtoms = np.zeros(py_totalAN, dtype=int)
        py_surfAtoms[:py_Nsurf] = surfatoms
        pos = self.get_positions()
        py_x, py_y, py_z = pos[:,0], pos[:,1], pos[:,2]
    
        # convert int to c_int
        totalAN = c_int(py_totalAN)
        Nsurf = c_int(py_Nsurf)
        # convert double to c_double
        distance = c_double(float(distance))
    
        #convert int array to c_int array
        surfAtoms = (c_int * py_totalAN)(*py_surfAtoms)
    
        # convert to c_double arrays
        x = (c_double * py_totalAN)(*py_x)
        y = (c_double * py_totalAN)(*py_y)
        z = (c_double * py_totalAN)(*py_z)
    
        sites = (c_double*(py_Nsurf * 3  ) )()
    
        _LIBCLUSGEO.getEta1.argtypes = [POINTER (c_double),POINTER (c_double), POINTER (c_double), POINTER (c_double), POINTER (c_int), c_int, c_int, c_double]
    
        _LIBCLUSGEO.getEta1(sites, x, y, z, surfAtoms, Nsurf, totalAN, distance)
    
        py_sites = np.ctypeslib.as_array( sites, shape=(py_Nsurf*3))
        py_sites= py_sites.reshape((py_Nsurf,3))
    
        return py_sites
    
    
    def _get_bridge_sites(self, distance = 1.8):
        """Takes an ASE atoms object,an array of surface atom indices and a distance as input
        Returns a 2D-array of top site positions with the defined distance from the surface atoms.
        """
        # get clusgeo internal format for c-code
        py_totalAN = len(self)
        surfatoms = self.get_surface_atoms()
        py_Nsurf = len(surfatoms)
        py_surfAtoms = np.zeros(py_totalAN, dtype=int)
        py_surfAtoms[:py_Nsurf] = surfatoms
        pos = self.get_positions()
        py_x, py_y, py_z = pos[:,0], pos[:,1], pos[:,2]
    
        # convert int to c_int
        totalAN = c_int(py_totalAN)
        Nsurf = c_int(py_Nsurf)
    
        #convert int array to c_int array
        surfAtoms = (c_int * py_totalAN)(*py_surfAtoms)
        # convert double to c_double
        distPy = distance
        distance = c_double(float(distance))
    
        # convert to c_double arrays
        x = (c_double * py_totalAN)(*py_x)
        y = (c_double * py_totalAN)(*py_y)
        z = (c_double * py_totalAN)(*py_z)
    
        sites = (c_double*(py_Nsurf * 3 * py_Nsurf  ) )()
    
    
        _LIBCLUSGEO.getEta2.argtypes = [POINTER (c_double),POINTER (c_double), POINTER (c_double), 
            POINTER (c_double), POINTER (c_int), c_int, c_int, c_double]
    
        Nbridge = _LIBCLUSGEO.getEta2(sites, x, y, z, surfAtoms, Nsurf, totalAN, distance)
    
        py_sites = np.ctypeslib.as_array( sites, shape=(py_Nsurf*3* py_Nsurf))
        py_sites = py_sites.reshape((py_Nsurf*py_Nsurf,3))
        py_sites = py_sites[:Nbridge] 
    
        # check whether adsorbate is inside
    
        full_ids = np.arange(py_totalAN)
        non_surfatoms = np.setdiff1d(full_ids, surfatoms, assume_unique = True)
        min_dist_inside_sites = np.min(cdist(pos[non_surfatoms], py_sites), axis = 0)
        min_dist_nonsurf_surf = np.min(cdist(pos[non_surfatoms], pos[surfatoms]))
        min_dist_all_sites = np.min(cdist(pos[full_ids], py_sites), axis = 0)
        outside_sites = py_sites[np.logical_and((min_dist_inside_sites > min_dist_nonsurf_surf), (min_dist_all_sites > (distPy - 0.1) ))]
        return outside_sites
    
    def _get_hollow_sites(self, distance= 1.8):
        """Takes an ASE atoms object, an array of surface atom indices and a distance as input
        Returns a 2D-array of top site positions with the defined distance from the surface atoms.
        """
        # get clusgeo internal format for c-code
        py_totalAN = len(self)
        surfatoms = self.get_surface_atoms()
        py_Nsurf = len(surfatoms)
        py_surfAtoms = np.zeros(py_totalAN, dtype=int)
        py_surfAtoms[:py_Nsurf] = surfatoms
        pos = self.get_positions()
        py_x, py_y, py_z = pos[:,0], pos[:,1], pos[:,2]
    
        # convert int to c_int
        totalAN = c_int(py_totalAN)
        Nsurf = c_int(py_Nsurf)
    
        #convert int array to c_int array
        surfAtoms = (c_int * py_totalAN)(*py_surfAtoms)
        # convert double to c_double
        distPy = distance
        distance = c_double(float(distance))
    
        # convert to c_double arrays
        x = (c_double * py_totalAN)(*py_x)
        y = (c_double * py_totalAN)(*py_y)
        z = (c_double * py_totalAN)(*py_z)
    
        sites = (c_double*(py_Nsurf * 3 * py_Nsurf  ) )()
    
    
        _LIBCLUSGEO.getEta3.argtypes = [POINTER (c_double),POINTER (c_double), POINTER (c_double), POINTER (c_double), 
            POINTER (c_int), c_int, c_int, c_double]
    
        Nhollow = _LIBCLUSGEO.getEta3(sites, x, y, z, surfAtoms, Nsurf, totalAN, distance)
    
        py_sites = np.ctypeslib.as_array( sites, shape=(py_Nsurf*3* py_Nsurf))
        py_sites = py_sites.reshape((py_Nsurf*py_Nsurf,3))
        py_sites = py_sites[:Nhollow] 
    
        # check whether adsorbate is inside
        full_ids = np.arange(py_totalAN)
        non_surfatoms = np.setdiff1d(full_ids, surfatoms, assume_unique = True)
        min_dist_inside_sites = np.min(cdist(pos[non_surfatoms], py_sites), axis = 0)
        min_dist_nonsurf_surf = np.min(cdist(pos[non_surfatoms], pos[surfatoms]))
        min_dist_all_sites = np.min(cdist(pos[full_ids], py_sites), axis = 0)
        outside_sites = py_sites[np.logical_and((min_dist_inside_sites > min_dist_nonsurf_surf), (min_dist_all_sites > (distPy - 0.1) ))]
    
        return outside_sites
    
    def get_sites(self, sitetype=-1,  distance= 1.8):
        if sitetype == -1:
            top = self._get_top_sites(distance=distance)
            bridge = self._get_bridge_sites(distance=distance)
            hollow = self._get_hollow_sites(distance=distance)
            return {1: top, 2: bridge, 3: hollow}
        elif sitetype == 1:
            return self._get_top_sites(distance=distance)
        elif sitetype == 2:
            return self._get_bridge_sites(distance=distance)
        elif sitetype == 3:
            return self._get_hollow_sites(distance=distance)
    
        else:
            raise ValueError("sitetype not understood. Use -1,1,2 or 3")
    

if __name__ == "__main__":
    #atoms = ase.io.read("h2o2.xyz")
    atoms = ase.io.read("BEAUT/beaut6.xyz")
    atoms = ase.io.read("tests/au40cu40.xyz")
    surfatoms = get_surface_atoms(atoms)

    print("surface atoms:", surfatoms.shape)
    surfit = atoms[surfatoms]
    #ase.io.write("surface.xyz", surfit)
    print("done get surface atoms")

    topHxyz = get_top_sites(atoms, surfatoms)
    print(topHxyz.shape)
    topH = ase.Atoms('H'*len(topHxyz), topHxyz)
    structure_topH = atoms + topH
    #ase.io.write("structure_topH.xyz", structure_topH)
    print("done get top sites")


    bridgeHxyz = get_bridge_sites(atoms, surfatoms)
    print(bridgeHxyz.shape)
    
    bridgeH = ase.Atoms('H'*len(bridgeHxyz), bridgeHxyz)
    structure_bridgeH = atoms + bridgeH
    #ase.io.write("structure_bridgeH.xyz", structure_bridgeH)
    print("done get bridge sites")


    hollowHxyz = get_hollow_sites(atoms, surfatoms)
    print(hollowHxyz.shape)
    
    hollowH = ase.Atoms('H'*len(hollowHxyz), hollowHxyz)
    structure_hollowH = atoms + hollowH
    #ase.io.write("structure_hollowH.xyz", structure_hollowH)
    print("done get hollow sites")
