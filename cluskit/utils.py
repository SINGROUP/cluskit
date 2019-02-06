import numpy as np 
import os, argparse, glob
from ctypes import *
from scipy.spatial.distance import cdist

_PATH_TO_CLUSGEO_SO = os.path.dirname(os.path.abspath(__file__))
_CLUSGEO_SOFILES = glob.glob( "".join([ _PATH_TO_CLUSGEO_SO, "/../lib/libcluskit3.*so*"]) )
_LIBCLUSGEO = CDLL(_CLUSGEO_SOFILES[0])


# currently not used, but might come in handy at some point
def _format_ase2cluskit(obj, all_atomtypes=[]):
    """ Takes an ase Atoms object and returns numpy arrays and integers
    which are read by the internal cluskit. Apos is currently a flattened
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


def get_adsorbate_vectors():
    pass


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



# needs to be updated, possibly without pathlib (python 2 ?!)
def write_all_sites(atoms, structure_name = "ref", path = "./"):
    """Takes an ASE atoms object and a string of the name of the structure.
    Returns None
    It writes xyzfiles with top, bridge and hollow sites into directories
    """
    surfatoms = get_surface_atoms(atoms)
    topHxyz = get_top_sites(atoms, surfatoms)
    edgeHxyz = get_edge_sites(atoms, surfatoms)
    hollowHxyz = get_hollow_sites(atoms, surfatoms)

    topH = ase.Atoms('H'*len(topHxyz), topHxyz)
    edgeH = ase.Atoms('H'*len(edgeHxyz), edgeHxyz)
    hollowH = ase.Atoms('H'*len(hollowHxyz), hollowHxyz)

    # write structures with all top, bridge and hollow sites


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

# TODO replace instances of self with arguments

def get_surface_atoms(self, bubblesize = 2.5, mask=False):
    """Determines the surface atoms of the nanocluster. Takes two optional inputs. 
    Firstly, a bubblesize determining how concave a surface can be (the default usually works well).
    Secondly, the boolean argument mask (default is False). 
    If set to True, a mask array will be returned.
    If False, it returns an array of indices of surface atoms.
    """
    if self.has('surface'):
        if mask:
            return self.arrays['surface']
        else:
            return np.nonzero(self.arrays['surface'])[0]

    # get cluskit internal format for c-code
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

    if mask:
        return surface_hold
    else:
        return py_surfAtoms

def get_nonsurface_atoms(self, bubblesize = 2.5):
    """ Determines the core / non-surface atoms of the nanocluster. 
    Takes a bubblesize as an input determining how concave a surface can be (the default usually works well).
    Returns an array of indices of surface atoms.
    
    """
    surface = self.get_surface_atoms(mask=True)
    return np.nonzero(np.logical_not(surface))[0]

def _get_top_sites(self, distance=1.5):
    """Takes a distance as input.
    Returns a 2D-array of top site positions with the defined distance from the adsorbing surface atom.
    """
    # get cluskit internal format for c-code
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
    self.site_positions[1] = py_sites
    return self.site_positions[1]    

def _get_bridge_sites(self, distance = 1.8):
    """Takes a distance as input.
    Returns a 2D-array of bridge site positions with the defined distance from the adsorbing surface atoms.
    """
    # get cluskit internal format for c-code
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
    self.site_positions[2] = outside_sites
    return self.site_positions[2]


def _get_hollow_sites(self, distance= 1.8):
    """Takes a distance as input.
    Returns a 2D-array of hollow site positions with the defined distance from the adsorbing surface atoms.
    """
    # get cluskit internal format for c-code
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
    self.site_positions[3] = outside_sites
    return self.site_positions[3]


if __name__ == "__main__":
    # coding: utf-8
    import ase
    from ase.visualize import view
    from ase.build import molecule
    from ase.cluster.icosahedron import Icosahedron
    import cluskit

    atoms = Icosahedron('Cu', noshells=3)
    cluster = cluskit.Cluster(atoms)

    zero_site = cluster.get_positions()[53]
    arbitrary_vector = [-2,-2,-2]
    adsorbate_x = ase.Atoms('HHCX', positions=[[2,0,0], [0,2,0], [0,0,0], [-1.4,-1.4, 0]])
    
    adsorbate = place_molecule_on_site(molecule = adsorbate_x, zero_site = zero_site, adsorption_vector = arbitrary_vector)

    # visualize
    clus_ads = cluster + adsorbate
    #view(clus_ads)
    adsorbate_vector_ase = ase.Atoms('OO', positions= [zero_site + arbitrary_vector, zero_site + np.multiply(arbitrary_vector, 2)])
    view(clus_ads + adsorbate_vector_ase)
