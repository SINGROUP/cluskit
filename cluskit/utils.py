import numpy as np 
import os, argparse, glob
from ctypes import *
from scipy.spatial.distance import cdist
import ase
from ase.constraints import FixAtoms, FixBondLengths, FixInternals, Hookean
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
from ase.neighborlist import NeighborList
import copy


_PATH_TO_CLUSGEO_SO = os.path.dirname(os.path.abspath(__file__))
_CLUSGEO_SOFILES = glob.glob( "".join([ _PATH_TO_CLUSGEO_SO, "/../lib/libcluskit3.*so*"]) )
_LIBCLUSGEO = CDLL(_CLUSGEO_SOFILES[0])

def x2_to_x(pos, bondlength):
    """Takes a 2D-array of adsorbed species positions and a bondlength as input.
    Returns adsorbed species positions where X2 molecules (distance lower than 
    bondlength) are replaced by X atoms

    Args:
        pos (2D ndarray) : positions of adsorbate atoms
        bondlength (float) :    bond length required to form a dimer    

    Returns:
        2D ndarray :    modified positions
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


def place_molecule_on_site(molecule, zero_site, adsorption_vector, remove_x = True):
    """Takes a molecule (ase object) as well as a zero site position with an adsorption vector as input.
    The object needs to contain an X (dummy atom) to mark the anchoring to the zero site.
    Returns a translated and rotated adsorbate (ase object) minus the anchor X.

    Args:
        molecule (ase.Atoms) :      instance of the molecules to be adsorbed
                                    needs to contain an anchor atom X
        zero_site (1D ndarray) :    position of the zerosite (on a top site, this is the cluster
                                    atom itself, on a bridge site it is the middle between the
                                    two cluster atoms, on a hollow site it is the geometrical
                                    center of the atomic triad)
        adsorption_vector (1D ndarray) :    adsorption vector of the site
        remove_x (bool) :   should the dummy atom be stripped from the 
                            atoms objects    
    Returns:
        ase.Atoms :     rotated and translated (possibly stripped of 'X' atom) 
                        copy of the molecule
    """    
    x_idx = np.where(molecule.get_atomic_numbers() == 0)[0]
    notx_idx = np.where(molecule.get_atomic_numbers() != 0)[0]

    pos = molecule.get_positions()
    dist_fromx = cdist(pos[x_idx], pos[notx_idx])
    connecting_atom = np.argmin(dist_fromx)
    connecting_atom_idx = notx_idx[connecting_atom]
    
    adsorbate = molecule.copy()
    adsorbate.translate( zero_site - adsorbate.get_positions()[x_idx] )
    adsorbate.rotate(adsorbate.get_positions()[connecting_atom_idx] - zero_site, 
        v=adsorption_vector, center=zero_site, rotate_cell=False)
    
    if remove_x == True:
        # remove dummy atom
        # only one X atom supported
        adsorbate.pop(x_idx[0])
    return adsorbate


def get_surface_atoms(atoms, bubblesize = 2.5, mask=False):
    """Determines the surface atoms of the nanocluster using a 
    ray bombarding algorithm (Dudenstein algorithm)
    Conceived and programmed by Eiaki Morooka.

    Args:
        atoms (ase.Atoms) : nanocluster structure
        bubblesize (float) :    Determines how concave a surface can be 
                                (the default usually works well)
        mask (bool) :   If set to True, a mask array will be returned.
                        If False, it returns an array of indices of 
                        surface atoms.

    Returns:
        1D ndarray :    atomic indices comprising the surface
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

    Args:
        atoms (ase.Atoms) : nanocluster structure
        bubblesize (float) :    Determines how concave a surface can be 
                                (the default usually works well)

    Returns:
        1D ndarray :    atomic indices comprising the core
    """
    surface = get_surface_atoms(atoms, bubblesize = bubblesize, 
        mask=True)
    return np.nonzero(np.logical_not(surface))[0]


def _get_top_sites(atoms, surfatoms, distance=1.5):
    """Takes an atoms object, surface atom indices and 
    optionally a distance as input.
    Returns a 2D-array of top site positions with the 
    defined distance from the adsorbing surface atom.

    This is an older algorithm which uses c-code. It works well,
    although not as robust as the delaunay algorithm

    Args:
        atoms (ase.Atoms) : nanocluster structure
        surfatoms (1D ndarray) :    atomic indices comprising the surface
        distance (float) :          distance from zerosite to adsorbate

    Returns:
        2D ndarray : positions of the top adsorption sites
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
    
    This is an older algorithm which uses c-code. It works well,
    although not as robust as the delaunay algorithm

    Args:
        atoms (ase.Atoms) : nanocluster structure
        surfatoms (1D ndarray) :    atomic indices comprising the surface
        distance (float) :          distance from zerosite to adsorbate

    Returns:
        2D ndarray : positions of the bridge adsorption sites
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

    This is an older algorithm which uses c-code. It works well,
    although not as robust as the delaunay algorithm

    Args:
        atoms (ase.Atoms) : nanocluster structure
        surfatoms (1D ndarray) :    atomic indices comprising the surface
        distance (float) :          distance from zerosite to adsorbate

    Returns:
        2D ndarray : positions of the hollow adsorption sites
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


def _filter_dihedrals(atoms, neighbor_lst, dihedrals,
    sp3_dct={'C': 4, 'N': 3, 'O': 2, 'S': 2, 'P': 3, 'Si': 4}):
    """Helper function for place_and_preoptimize_adsorbates
    
    Filters out all dihedrals which do not have 2 sp3-centers
    on positions 2 and 3 of the dihedral.
    Args:
        atoms (ase.Atoms)   :   adsorbate molecule
        nb_lst (list)  :    tuples of bond distances, 
                            pairs of atom indices
        dihedrals (list) :  tuples of dihedral values and their
                            quadruplets of indices
        sp3_dct (dict) :    determines at how many neigbors an
                            atom becomes an sp3-center

    Returns:
        list :  kept dihedrals in the same format as dihedrals
    """
    kept_dihedrals = []
    for value, dihedral in dihedrals:
        j, k = dihedral[1], dihedral[2]
        j_nb, _ = neighbor_lst.get_neighbors(j)
        k_nb, _ = neighbor_lst.get_neighbors(k)
        j_nn = len(j_nb)
        k_nn = len(k_nb)
        j_flex = sp3_dct[atoms.get_chemical_symbols()[j]]
        k_flex = sp3_dct[atoms.get_chemical_symbols()[k]]

        if (j_nn < j_flex) or (k_nn < k_flex):
            kept_dihedrals.append([value, dihedral])
    return kept_dihedrals


def _determine_cutoffs(atoms, buffer_factor=1.0,
    radii_dct={'H': 0.37, 'C': 0.77, 'N': 0.75, 'O': 0.73,
             'F': 0.71, 'B': 0.88, 'P': 1.036, 'S': 1.02,
             'X': 0.77}, from_ase = True):
    """Helper function for place_and_preoptimize_adsorbates

    Args:
        atoms (ase.Atoms)   :   adsorbate molecule
        buffer_factor (float) : factor by which the atomic radii get scaled
        radii_dct (dict) :      atomic radii to determine neighbors. Is 
                                ignored if from_ase is True
        from_ase (bool) :   If set to True, ase makes an educated guess
                            of the neigbors using sensible atomic radii

    Returns:
        list : cutoff (float) for each atom in atoms
    """
    if from_ase:
        cutoffs = ase.utils.natural_cutoffs(atoms, mult=buffer_factor)
    else:
        ch_sym = atoms.get_chemical_symbols()
        cutoffs = [buffer_factor * radii_dct[symbol] for symbol in ch_sym]
    return cutoffs

def _get_neighbours(atoms, buffer_factor = 1.0,
    radii_dct ={'H': 0.37, 'C': 0.77, 'N': 0.75, 'O': 0.73,
               'F': 0.71, 'B': 0.88, 'P': 1.036, 'S': 1.02,
               'X': 0.77}, from_ase = True):
    """Helper function for place_and_preoptimize_adsorbates

    Args:
        atoms (ase.Atoms)   :   adsorbate molecule
        buffer_factor (float) : factor by which the atomic radii get scaled
        radii_dct (dict) :      atomic radii to determine neighbors. Is 
                                ignored if from_ase is True
        from_ase (bool) :   If set to True, ase makes an educated guess
                            of the neigbors using sensible atomic radii

    Returns:
        tuple :     list of neigbours (tuples of bond distances and pairs of atom indices),
                    ase.neighborlist.NeighborList object
    """
    cutoffs = _determine_cutoffs(atoms, buffer_factor=buffer_factor, radii_dct = radii_dct, from_ase = from_ase)
    nl = NeighborList(cutoffs=cutoffs, bothways=True, self_interaction=False, skin=0.0)
    nl.update(atoms)
    cmat = nl.get_connectivity_matrix(sparse=False)
    cmat = np.triu(cmat)
    nb_lst = np.transpose(np.nonzero(cmat))
    return nb_lst, nl

def _fix_nearest_atom_to_x(atoms, nl):
    """Helper function for place_and_preoptimize_adsorbates

    Fixes the position of the atom closest to the dummy atom
    X.

    Args:
        atoms (ase.Atoms)   :   adsorbate molecule
        nl (ase.neighborlist.NeighborList)  : neighborlist object

    Returns:
        ase.constraints.FixAtoms :  constraint to fix the 
                                    absolute position of one atom
    """
    x_idx = [atom.index for atom in atoms if atom.symbol == 'X'][0]
    first_atom, offset = nl.get_neighbors(x_idx)
    #print(first_atom)
    fa = FixAtoms(first_atom)
    return fa

def _get_all_angles(atoms, nb_lst):
    """Helper function for place_and_preoptimize_adsorbates
   
    Args:
        atoms (ase.Atoms)   :   adsorbate molecule
        nb_lst (list)  :    tuples of bond distances, 
                            pairs of atom indices

    Returns:
        list :      all angles which can be constructed
                    from two neighbor pairs
    """
    angles = []
    for first_pair in nb_lst:
        for second_pair in nb_lst:
            i, j = first_pair
            k, l = second_pair
            if (i < k) or (j < l):
                pass
            elif len(set([i,j,k,l])) == 3:
                if i == k:
                    angle = j, i, l
                if i == l:
                    angle = j, i, k
                if j == k:
                    angle = i, j, l
                if j == l:
                    angle = i, j, k
                value = atoms.get_angle(*angle) * np.pi / 180
                angles.append([value, angle])
    return angles

def _get_all_dihedrals(atoms, angles):
    """Helper function for place_and_preoptimize_adsorbates
    
    Args:
        atoms (ase.Atoms)   :   adsorbate molecule
        angles (list)  :    tuples of angles, indices 
                            within the molecule

    Returns:
        list :      all dihedrals which can be constructed
                    from two angles
    """
    dihedrals = []
    for idx1 ,(_, angle1) in enumerate(angles):
        for idx2 ,(_, angle2) in enumerate(angles):
            if idx1 < idx2:
                continue
            i1 ,j1, k1 = angle1[0], angle1[1], angle1[2]
            i2, j2, k2 = angle2[0], angle2[1], angle2[2]
            if len(set([i1, j1, k1, i2, j2, k2])) == 4:
                if (j1 == j2):
                    pass
                elif ((j1 == i2) or (j1 == k2)) and ((j2 == i1) or (j2 == k1)):
                    dihedral = [j1, j2]
                    if len(set([i1, j1, j2])) == 3:
                        dihedral.insert(0, i1)
                    else:
                        dihedral.insert(0, k1)
                    if len(set([k2, j1, j2])) == 3:
                        dihedral.append(k2)
                    else:
                        dihedral.append(i2)
                    value = atoms.get_dihedral(*dihedral) * np.pi / 180
                    dihedrals.append([value, dihedral])
    return dihedrals

def _constrain_molecule(atoms, rattle=0.00001):
    """Helper function for place_and_preoptimize_adsorbates
    
    Args:
        atoms (ase.Atoms)   :   adsorbate molecule
        rattle (float)  : standard deviation of the structure rattling

    Returns:
        list :      list of different constraints from ase.constraints 
                    bonds, angles and dihedrals other than those of
                    sp3-centers are constrained
    """
    # Rattling is important, otherwise the optimization fails!
    atoms.rattle(stdev=rattle)

    nb_lst, nl = _get_neighbours(atoms, buffer_factor=1.5)
    fb = FixBondLengths(nb_lst)
    fa = _fix_nearest_atom_to_x(atoms, nl)

    angles = _get_all_angles(atoms, nb_lst)
    dihedrals = _get_all_dihedrals(atoms, angles)
    # keep dihedrals with sp2 or sp center
    # use neighbor list and a custom dictionary for number of nearest neighbors
    dihedrals = _filter_dihedrals(atoms, nl, dihedrals)
    print("kept dhls", dihedrals)

    fi = FixInternals(angles= angles, dihedrals = dihedrals)

    hookean_lst = _hookean_bonds(atoms, nb_lst)

    # IMPORTANT! fix binding atom at the end, otherwise it moves!
    atoms.set_constraint(hookean_lst.extend([fi, fb, fa]))
    return atoms

def _hookean_bonds(atoms, nb_lst):
    """Helper function for place_and_preoptimize_adsorbates.

    Args:
        atoms (ase.Atoms)   :   adsorbate molecule
        nb_lst (ase.neighborlist.NeighborList)  : neighborlist

    Returns:
        ase.constraints.Hookean :   Hookean bond contraints of all bonds
                                    in the molecule
    """
    hookean_lst = []
    for neighbors in nb_lst:
        c = Hookean(a1=int(neighbors[0]), a2=int(neighbors[1]), rt=atoms.get_distance(neighbors[0], neighbors[1]), k=20.)
        hookean_lst.append(c)
    return hookean_lst


def place_and_preoptimize_adsorbates(cluster, molecule, sitetype = -1, max_distance = 3.5,
    n_remaining = 100, is_reduce = True, is_reset = False, n_lj_steps = 100):
    """
    This function first places adsorbates on the surface of a nanocluster
    Then, the positions are preoptimized with a Lennard-Jones
    potential.
    If reduced is set to True, adsorbates are removed one-by-one
    intermittently optimizing the structure until no more
    adsorbates need to be removed.
    Use the parameters
    -n_remaining
    -max_distance
    to determine when removing adsorbates ought to stop.

    Args:
        cluster (cluskit.Cluster) : nanocluster against which adsorbates
                                    are placed
        molecule (ase.Atoms) :      instance of the molecules to be adsorbed
                                    needs to contain an anchor atom X
        sitetype (int)      :   1 : "top", 2 : "bridge", 3 : "hollow" site
                                -1 : all of the above 
        max_distance (float) :  the adsorbates will be seperated by at least
                                this distance in Angstrom. Reduces number of
                                adsorbates.
        n_remaining (int)   :   The number of adsorbates are reduced eliminating
                                the closest ones to each other.
        is_reduce (bool)    :   allows for reduction with the criteria 'max_distance'
                                or 'n_remaining'. The search is complete when one
                                criterion is met.
        is_reset (bool)     :   if set to True, the initial positions of the adsorbates
                                are resumed after an adsorbate has been eliminated
        n_lj_steps (int)    :   number of steps to iterate in the Lennard-Jones 
                                optimizer

    Returns:
        list :  adsorbates in relative position to the surface of
                the cluster
    """
    # set constraints in molecule
    molecule = _constrain_molecule(molecule, rattle=0.00001)
    n_atoms_per_molecule = len(molecule)

    # make a whole bunch of adsorbates on the surface of a nanocluster
    # initial guess orientation
    adsorbates = cluster.place_adsorbates(molecule, sitetype=sitetype, remove_x = False)
    combined_adsorbates = adsorbates[0]
    for ads in adsorbates[1:]:
        combined_adsorbates += ads

    atoms = combined_adsorbates

    # loop
    for z in range(10000):
        # save initial placement
        initial_adsorbates = copy.deepcopy(atoms)
        pos_initial = atoms.get_positions()

        ## Optimize structure
        # LJ
        # set sigma high for repulsive interactions
        atoms.set_calculator(LennardJones(epsilon = 1e-10 , sigma = 5.0, rc = 10.0))
        dyn = BFGS(atoms)
        dyn.run(fmax=0.1, steps=n_lj_steps)

        pos_final = atoms.get_positions()
        #print(pos_final - pos_initial)

        # compute distances
        n_adsorbates = int(len(atoms) / n_atoms_per_molecule)
        print("number of adsorbates:", n_adsorbates)

        if is_reduce == False:
            break

        min_dist = 10000.0
        for i in range(n_adsorbates-1):
            adsorbate_pos = atoms.get_positions()[i*n_atoms_per_molecule:(i+1)*n_atoms_per_molecule]
            rest_pos = atoms.get_positions()[(i+1)*n_atoms_per_molecule:]
            dmat = cdist(adsorbate_pos, rest_pos)
            min_dist_arr = dmat.min(axis=1)
            min_dist_x = min_dist_arr.min()
            min_idx = min_dist_arr.argmin()
            if min_dist_x < min_dist:
                min_dist = min_dist_x
                ads_idx = int((min_idx + (i+1)*(n_atoms_per_molecule)) / n_atoms_per_molecule)
                ads_idx2 = i

        # additional check, which of the two adsorbates should be removed
        mask = np.ones(len(atoms), dtype=bool)
        mask[ads_idx * n_atoms_per_molecule:(ads_idx + 1) * n_atoms_per_molecule] = 0
        mask[ads_idx2 * n_atoms_per_molecule:(ads_idx2 + 1) * n_atoms_per_molecule] = 0

        adsorbate_pos1 = atoms.get_positions()[ads_idx * n_atoms_per_molecule:(ads_idx + 1) * n_atoms_per_molecule]
        adsorbate_pos2 = atoms.get_positions()[ads_idx2 * n_atoms_per_molecule:(ads_idx2 + 1) * n_atoms_per_molecule]
        rest_pos = atoms.get_positions()[mask]

        dmat1 = cdist(adsorbate_pos1, rest_pos)
        dmat2 = cdist(adsorbate_pos2, rest_pos)
        min_dist1 = dmat1.min(axis=1).min()
        min_dist2 = dmat2.min(axis=1).min()

        if min_dist1 < min_dist2:
            ads_idx = ads_idx
        else:
            ads_idx = ads_idx2

        # check stop criteria
        if (n_adsorbates <= n_remaining):
            print(str(n_remaining) + "adsorbates remain on the surface. Concluding...")
            break
        elif min_dist > max_distance:
            print("minimum distance between adsorbates is", min_dist, ". Concluding...")
            break
        else:
            # remove closest adsorbate
            # update adsorbate list
            mask = np.ones(len(initial_adsorbates), dtype=bool)
            mask[ads_idx*n_atoms_per_molecule:(ads_idx+1)*n_atoms_per_molecule] = 0
            if is_reset:
                atoms = initial_adsorbates[mask]
            else:
                atoms = atoms[mask]
        ## end loop

    # remove x's
    adsorbate_lst = []
    for i in range(n_adsorbates):
        adsorbate_x = atoms[i*n_atoms_per_molecule:(i+1)*n_atoms_per_molecule]
        x_idx = [atom.index for atom in adsorbate_x if atom.symbol == 'X'][0]
        adsorbate_x.pop(x_idx)
        adsorbate = adsorbate_x
        adsorbate_lst.append(adsorbate)
    return adsorbate_lst
