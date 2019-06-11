import numpy as np 
import os, argparse, glob
from ctypes import *
from scipy.spatial.distance import cdist

_PATH_TO_CLUSGEO_SO = os.path.dirname(os.path.abspath(__file__))
_CLUSGEO_SOFILES = glob.glob( "".join([ _PATH_TO_CLUSGEO_SO, "/../lib/libcluskit3.*so*"]) )
_LIBCLUSGEO = CDLL(_CLUSGEO_SOFILES[0])

def x2_to_x(pos, bondlength):
    """Takes a 2D-array of adsorbed species positions and a bondlength as input.
    Returns adsorbed species positions where X2 molecules (distance lower than bondlength) are replaced by X atoms
    """
    py_totalAN = pos.shape[0]
    py_x, py_y, py_z = pos[:,0], pos[:,1], pos[:,2]
    updated_pos = np.zeros(pos.shape)
    py_x_new, py_y_new, py_z_new, = updated_pos[:,0], updated_pos[:,1], updated_pos[:,2]

    # currently only implemented for single type adsorbate
    atomtype_lst = [1] 
    py_typeNs =  [py_totalAN]

    # convert int to c_int
    totalAN = c_int(py_totalAN)
    # convert double to c_double
    bondlength = c_double(float(bondlength))

    #convert int array to c_int array
    types = (c_int * len(atomtype_lst))(*atomtype_lst)
    typeNs = (c_int * len(py_typeNs))(*py_typeNs)

    # convert to c_double arrays
    x = (c_double * py_totalAN)(*py_x)
    y = (c_double * py_totalAN)(*py_y)
    z = (c_double * py_totalAN)(*py_z)

    x_new = (c_double * py_totalAN)(*py_x_new)
    y_new = (c_double * py_totalAN)(*py_y_new)
    z_new = (c_double * py_totalAN)(*py_z_new)


    _LIBCLUSGEO.x2_to_x.argtypes = [POINTER (c_double),POINTER (c_double), POINTER (c_double), POINTER (c_double),  
        POINTER (c_double), POINTER (c_double), POINTER (c_int), POINTER (c_int), c_double]
    _LIBCLUSGEO.x2_to_x.restype = c_int

    updated_totalAN = _LIBCLUSGEO.x2_to_x(x_new, y_new, z_new, x, y, z, types, typeNs, bondlength)

    py_x_new = np.ctypeslib.as_array(x_new, shape=(py_totalAN))
    py_y_new = np.ctypeslib.as_array(y_new, shape=(py_totalAN))
    py_z_new = np.ctypeslib.as_array(z_new, shape=(py_totalAN))

    updated_pos = np.array([py_x_new, py_y_new, py_z_new]).T

    updated_pos = updated_pos[:updated_totalAN]

    return updated_pos

def place_molecule_on_site(molecule, zero_site, adsorption_vector):
    """Takes a molecule (ase object) as well as a zero site position with an adsorption vector as input.
    The object needs to contain an X (dummy atom) to mark the anchoring to the zero site.
    Returns a translated and rotated adsorbate (ase object) minus the anchor X.
    """
    
    x_idx = np.where(molecule.get_atomic_numbers() == 0)[0]
    notx_idx = np.where(molecule.get_atomic_numbers() != 0)[0]

    pos = molecule.get_positions()
    dist_fromx = cdist(pos[x_idx], pos[notx_idx])
    connecting_atom = np.argmin(dist_fromx)
    connecting_atom_idx = notx_idx[connecting_atom]
    
    adsorbate = molecule.copy()
    adsorbate.translate( zero_site - adsorbate.get_positions()[x_idx] )
    adsorbate.rotate(adsorbate.get_positions()[connecting_atom_idx] - zero_site, v=adsorption_vector, center=zero_site, rotate_cell=False)
    
    # remove dummy atom
    # only one X atom supported
    adsorbate.pop(x_idx[0])

    return adsorbate


def get_surface_atoms(atoms, bubblesize = 2.5, mask=False):
    """Determines the surface atoms of the nanocluster. Takes two optional inputs. 
    Firstly, a bubblesize determining how concave a surface can be (the default usually works well).
    Secondly, the boolean argument mask (default is False). 
    If set to True, a mask array will be returned.
    If False, it returns an array of indices of surface atoms.
    """

    # get cluskit internal format for c-code
    py_totalAN = len(atoms)
    py_surfAtoms = np.zeros(py_totalAN, dtype=int)
    pos = atoms.get_positions()
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

    surface_hold = np.zeros(len(atoms), dtype='bool')
    surface_hold[py_surfAtoms] = True

    if mask:
        return surface_hold
    else:
        return py_surfAtoms



def get_nonsurface_atoms(atoms, bubblesize = 2.5):
    """ Determines the core / non-surface atoms of the nanocluster. 
    Takes a bubblesize as an input determining how concave a surface can be (the default usually works well).
    Returns an array of indices of surface atoms.
    
    """
    surface = get_surface_atoms(atoms, bubblesize = bubblesize, 
        mask=True)
    return np.nonzero(np.logical_not(surface))[0]


def _get_top_sites(atoms, surfatoms, distance=1.5):
    """Takes an atoms object, surface atom indices and 
    optionally a distance as input.
    Returns a 2D-array of top site positions with the 
    defined distance from the adsorbing surface atom.
    """
    # get cluskit internal format for c-code
    py_totalAN = len(atoms)
    #surfatoms = get_surface_atoms(atoms)
    py_Nsurf = len(surfatoms)
    py_surfAtoms = np.zeros(py_totalAN, dtype=int)
    py_surfAtoms[:py_Nsurf] = surfatoms
    pos = atoms.get_positions()
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

def _get_bridge_sites(atoms, surfatoms, distance = 1.8):
    """Takes an atoms object, surface atom indices and 
    optionally a distance as input.
    Returns a 2D-array of bridge site positions with the defined 
    distance from the adsorbing surface atoms.
    """
    # get cluskit internal format for c-code
    py_totalAN = len(atoms)
    py_Nsurf = len(surfatoms)
    py_surfAtoms = np.zeros(py_totalAN, dtype=int)
    py_surfAtoms[:py_Nsurf] = surfatoms
    pos = atoms.get_positions()
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


def _get_hollow_sites(atoms, surfatoms, distance= 1.8):
    """Takes an atoms object, surface atom indices and 
    optionally a distance as input.
    Returns a 2D-array of hollow site positions with the 
    defined distance from the adsorbing surface atoms.
    """
    # get cluskit internal format for c-code
    py_totalAN = len(atoms)
    py_Nsurf = len(surfatoms)
    py_surfAtoms = np.zeros(py_totalAN, dtype=int)
    py_surfAtoms[:py_Nsurf] = surfatoms
    pos = atoms.get_positions()
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

