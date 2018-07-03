import numpy as np 
import ase, ase.io
import os, argparse, glob
from ctypes import *
import pathlib

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

def get_surface_atoms(obj, bubblesize = 2.5):
    """Takes an ASE atoms object and a bubblesize determining how concave a surface can be.
    Returns an array of indices of surface atoms.
    """
    # get clusgeo internal format for c-code
    py_totalAN = len(obj)
    py_surfAtoms = np.zeros(py_totalAN, dtype=int)
    pos = obj.get_positions()
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

    #path_to_so = os.path.dirname(os.path.abspath(__file__))
    #sofiles = glob.glob( "".join([ path_to_so, "/../lib/libclusgeo3.*so*"]) )
    #libclusgeo = CDLL(sofiles[0])
    #libclusgeo.findSurf.argtypes = [POINTER (c_double),POINTER (c_double), POINTER (c_double), POINTER (c_int), c_int, c_double]
    #libclusgeo.findSurf.restype = c_int
    _LIBCLUSGEO.findSurf.argtypes = [POINTER (c_double),POINTER (c_double), POINTER (c_double), POINTER (c_int), c_int, c_double]
    _LIBCLUSGEO.findSurf.restype = c_int

    #Nsurf = libclusgeo.findSurf(x, y, z, surfAtoms, totalAN, bubblesize)
    Nsurf = _LIBCLUSGEO.findSurf(x, y, z, surfAtoms, totalAN, bubblesize)

    py_surfAtoms = np.ctypeslib.as_array( surfAtoms, shape=(py_totalAN))
    py_surfAtoms = py_surfAtoms[:Nsurf]

    return py_surfAtoms
####### NEW 2018/Jun/26 - Aki########################################################
#def get_nonsurf_atoms(obj, bubblesize = 2.5):
#    """Takes an ASE atoms object and a bubblesize determining how concave a surface can be.
#    Returns an array of indices of surface atoms.
#    """
#    # get clusgeo internal format for c-code
#    py_totalAN = len(obj)
#    py_surfAtoms = np.zeros(py_totalAN, dtype=int)
#    py_nonsurfAtoms = np.zeros(py_totalAN, dtype=int) #++
#    pos = obj.get_positions()
#    py_x, py_y, py_z = pos[:,0], pos[:,1], pos[:,2]
#
#    # convert int to c_int
#    totalAN = c_int(py_totalAN)
#    # convert double to c_double
#    bubblesize = c_double(float(bubblesize))
#    #convert int array to c_int array
#    surfAtoms = (c_int * py_totalAN)(*py_surfAtoms)
#    nonSurf = (c_int * py_totalAN)(*py_nonsurfAtoms)
#
#    # convert to c_double arrays
#    x = (c_double * py_totalAN)(*py_x)
#    y = (c_double * py_totalAN)(*py_y)
#    z = (c_double * py_totalAN)(*py_z)
#
#    _LIBCLUSGEO.findSurf.argtypes = [POINTER (c_double),POINTER (c_double), POINTER (c_double), POINTER (c_int), c_int, c_double]
#    _LIBCLUSGEO.getNonSurf.argtypes = [POINTER (c_double), c_int, c_int, POINTER (c_double)]
#
#    _LIBCLUSGEO.findSurf.restype = c_int
#    _LIBCLUSGEO.getNonSurf.restype = c_int
#
#    Nsurf = _LIBCLUSGEO.findSurf(x, y, z, surfAtoms, totalAN, bubblesize) 
#    NnonSurf = _LIBCLUSGEO.getNonSurf(nonSurf, totalAN, Nsurf, surfAtoms) 
#    
#
##    py_surfAtoms = np.ctypeslib.as_array( surfAtoms, shape=(py_totalAN))
##    py_surfAtoms = py_surfAtoms[:Nsurf]
#    py_nonsurfAtoms = np.ctypeslib.as_array( nonSurf, shape=(py_totalAN))
#    py_surfAtoms = py_surfAtoms[:NnonSurf]
#
#    return py_nonsurfAtoms
#
#################################################################################################
def get_top_sites(obj, surfatoms, distance=1.5):
    """Takes an ASE atoms object, an array of surface atom indices and a distance as input
    Returns a 2D-array of top site positions with the defined distance from the surface.
    """
    # get clusgeo internal format for c-code
    py_totalAN = len(obj)
    py_Nsurf = len(surfatoms)
    py_surfAtoms = np.zeros(py_totalAN, dtype=int)
    py_surfAtoms[:py_Nsurf] = surfatoms
    pos = obj.get_positions()
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

    surfH = (c_double*(py_Nsurf * 3  ) )()

    _LIBCLUSGEO.getEta1.argtypes = [POINTER (c_double),POINTER (c_double), POINTER (c_double), POINTER (c_double), POINTER (c_int), c_int, c_int, c_double]

    _LIBCLUSGEO.getEta1(surfH, x, y, z, surfAtoms, Nsurf, totalAN, distance)

    py_surfH = np.ctypeslib.as_array( surfH, shape=(py_Nsurf*3))
    py_surfH = py_surfH.reshape((py_Nsurf,3))

    return py_surfH


def get_edge_sites(obj, surfatoms, distance = 1.8):
    """Takes an ASE atoms object,an array of surface atom indices and a distance as input
    Returns a 2D-array of top site positions with the defined distance from the surface atoms.
    """
    # get clusgeo internal format for c-code
    py_totalAN = len(obj)
    py_Nsurf = len(surfatoms)
    py_surfAtoms = np.zeros(py_totalAN, dtype=int)
    py_surfAtoms[:py_Nsurf] = surfatoms
    pos = obj.get_positions()
    py_x, py_y, py_z = pos[:,0], pos[:,1], pos[:,2]

    # convert int to c_int
    totalAN = c_int(py_totalAN)
    Nsurf = c_int(py_Nsurf)

    #convert int array to c_int array
    surfAtoms = (c_int * py_totalAN)(*py_surfAtoms)
    # convert double to c_double
    distance = c_double(float(distance))

    # convert to c_double arrays
    x = (c_double * py_totalAN)(*py_x)
    y = (c_double * py_totalAN)(*py_y)
    z = (c_double * py_totalAN)(*py_z)

    surfH = (c_double*(py_Nsurf * 3 * py_Nsurf  ) )()


    _LIBCLUSGEO.getEta2.argtypes = [POINTER (c_double),POINTER (c_double), POINTER (c_double), 
        POINTER (c_double), POINTER (c_int), c_int, c_int, c_double]

    Nedge = _LIBCLUSGEO.getEta2(surfH, x, y, z, surfAtoms, Nsurf, totalAN, distance)

    py_surfH = np.ctypeslib.as_array( surfH, shape=(py_Nsurf*3* py_Nsurf))
    py_surfH = py_surfH.reshape((py_Nsurf*py_Nsurf,3))
    py_surfH = py_surfH[:Nedge] 

    return py_surfH

def get_hollow_sites(obj, surfatoms, distance= 1.8):
    """Takes an ASE atoms object, an array of surface atom indices and a distance as input
    Returns a 2D-array of top site positions with the defined distance from the surface atoms.
    """
    # get clusgeo internal format for c-code
    py_totalAN = len(obj)
    py_Nsurf = len(surfatoms)
    py_surfAtoms = np.zeros(py_totalAN, dtype=int)
    py_surfAtoms[:py_Nsurf] = surfatoms
    pos = obj.get_positions()
    py_x, py_y, py_z = pos[:,0], pos[:,1], pos[:,2]

    # convert int to c_int
    totalAN = c_int(py_totalAN)
    Nsurf = c_int(py_Nsurf)

    #convert int array to c_int array
    surfAtoms = (c_int * py_totalAN)(*py_surfAtoms)
    # convert double to c_double
    distance = c_double(float(distance))

    # convert to c_double arrays
    x = (c_double * py_totalAN)(*py_x)
    y = (c_double * py_totalAN)(*py_y)
    z = (c_double * py_totalAN)(*py_z)

    surfH = (c_double*(py_Nsurf * 3 * py_Nsurf  ) )()


    _LIBCLUSGEO.getEta3.argtypes = [POINTER (c_double),POINTER (c_double), POINTER (c_double), POINTER (c_double), 
        POINTER (c_int), c_int, c_int, c_double]

    Nhollow = _LIBCLUSGEO.getEta3(surfH, x, y, z, surfAtoms, Nsurf, totalAN, distance)

    py_surfH = np.ctypeslib.as_array( surfH, shape=(py_Nsurf*3* py_Nsurf))
    py_surfH = py_surfH.reshape((py_Nsurf*py_Nsurf,3))
    py_surfH = py_surfH[:Nhollow] 

    return py_surfH


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
    print(updated_totalAN)

    return updated_pos



def write_all_sites(atoms, structure_name = "ref", path = "./"):
    """Takes an ASE atoms object and a string of the name of the structure.
    Returns None
    It writes xyzfiles with top, edge and hollow sites into directories
    """
    surfatoms = get_surface_atoms(atoms)
    topHxyz = get_top_sites(atoms, surfatoms)
    edgeHxyz = get_edge_sites(atoms, surfatoms)
    hollowHxyz = get_hollow_sites(atoms, surfatoms)

    topH = ase.Atoms('H'*len(topHxyz), topHxyz)
    edgeH = ase.Atoms('H'*len(edgeHxyz), edgeHxyz)
    hollowH = ase.Atoms('H'*len(hollowHxyz), hollowHxyz)

    # write structures with all top, edge and hollow sites


    pathlib.Path("top").mkdir(parents=True, exist_ok=True) 
    pathlib.Path("edge").mkdir(parents=True, exist_ok=True) 
    pathlib.Path("hollow").mkdir(parents=True, exist_ok=True) 

    structure_topH = atoms + topH
    ase.io.write("top/structure_topH.xyz", structure_topH)
    structure_edgeH = atoms + edgeH
    ase.io.write("edge/structure_edgeH.xyz", structure_edgeH)
    structure_hollowH = atoms + hollowH
    ase.io.write("hollow/structure_hollowH.xyz", structure_hollowH)

    # write into files with single hydrogen
    for adatom, idx in zip(topH, range(len(topH))):
        #print(adatom)
        structure_H = atoms + adatom
        dirname = "top/" + "H" + str(idx + 1)
        pathlib.Path(dirname).mkdir(parents=True, exist_ok=True) 
        ase.io.write(dirname + "/" + structure_name + "H.xyz", structure_H)

    for adatom, idx in zip(edgeH, range(len(edgeH))):
        #print(adatom)
        structure_H = atoms + adatom
        dirname = "edge/" + "H" + str(idx + 1)
        pathlib.Path(dirname).mkdir(parents=True, exist_ok=True) 
        ase.io.write(dirname + "/" + structure_name + "H.xyz", structure_H)

    for adatom, idx in zip(hollowH, range(len(hollowH))):
        #print(adatom)
        structure_H = atoms + adatom
        dirname = "hollow/" + "H" + str(idx + 1)
        pathlib.Path(dirname).mkdir(parents=True, exist_ok=True) 
        ase.io.write(dirname + "/" + structure_name + "H.xyz", structure_H)       


    return None


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


    edgeHxyz = get_edge_sites(atoms, surfatoms)
    print(edgeHxyz.shape)
    
    #np.savetxt("edgeHxyz.txt",edgeHxyz)
    edgeH = ase.Atoms('H'*len(edgeHxyz), edgeHxyz)
    structure_edgeH = atoms + edgeH
    #ase.io.write("structure_edgeH.xyz", structure_edgeH)
    print("done get edge sites")


    hollowHxyz = get_hollow_sites(atoms, surfatoms)
    print(edgeHxyz.shape)
    
    #np.savetxt("edgeHxyz.txt",edgeHxyz)
    hollowH = ase.Atoms('H'*len(hollowHxyz), hollowHxyz)
    structure_hollowH = atoms + hollowH
    #ase.io.write("structure_hollowH.xyz", structure_hollowH)
    print("done get hollow sites")
    
    #write_all_sites(atoms, "au40cu40")
    surfatoms = get_surface_atoms(atoms)
    topHxyz = get_top_sites(atoms, surfatoms)
    updated_pos = x2_to_x(topHxyz, 3.0)

    print("Before x2 elimination", topHxyz.shape)
    print("After x2 elimination", updated_pos.shape)


