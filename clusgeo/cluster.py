import numpy as np 
import ase, ase.io
import os, argparse, glob
from ctypes import *
from scipy.spatial.distance import cdist, squareform, pdist

import dscribe
import soaplite

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


def _fps(pts, K, greedy=False):
    dist_matrix = squareform(pdist(pts))
    fts_ids = np.zeros(K, dtype='int')
     
    #choosing random start point
    fts_ids[0] = np.random.choice(pts.shape[0])
     
    #finding next k-1
    if greedy:
        for i in range(1, K):
            fts_ids[i] = np.argmax(dist_matrix[fts_ids[i-1]])
    else:
        min_dist = dist_matrix[fts_ids[0]]
        for i in range(1, K):
            fts_ids[i] = np.argmax(min_dist)
            min_dist = np.minimum(min_dist, dist_matrix[fts_ids[i]])
     
    return fts_ids

def _safe_fps(pts, K, greedy=False):
    dist_matrix = squareform(pdist(pts))
    fts_ids = np.zeros(K, dtype='int') -1
     
    #choosing random start point
    fts_ids[0] = np.random.choice(pts.shape[0])
     
    #finding next k-1
    if greedy:
        for i in range(1, K):
            arg = np.argmax(dist_matrix[fts_ids[i-1]])
            if arg in fts_ids:
                fts_ids[i] = np.argmax(np.isin(np.arange(K), fts_ids, invert=True))
            else:
                fts_ids[i] = np.argmax(dist_matrix[fts_ids[i-1]])
    else:
        min_dist = dist_matrix[fts_ids[0]]
        for i in range(1, K):
            if min_dist.max() < 10e-10:
                fts_ids[i] = np.argmax(np.isin(np.arange(K), fts_ids, invert=True))
            else:
                fts_ids[i] = np.argmax(min_dist)
                min_dist = np.minimum(min_dist, dist_matrix[fts_ids[i]])   
    return fts_ids

def _rank_fps(pts, K, greedy=False, is_safe = False):
    if is_safe:
        ranked_lst = _safe_fps(pts, K, greedy = greedy)
    else:
        ranked_lst = _fps(pts, K, greedy = greedy)

    return ranked_lst




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

        self.site_positions = {}
        self.cluster_descriptor = None
        self.sites_descriptor = {}






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
        self.site_positions[1] = py_sites
        return self.site_positions[1]    
    
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
        self.site_positions[2] = outside_sites
        return self.site_positions[2]
    
    
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
        self.site_positions[3] = outside_sites
        return self.site_positions[3]
    
    def get_sites(self, sitetype=-1,  distance= 1.8):
        if sitetype == -1:
            top = self._get_top_sites(distance=distance)
            bridge = self._get_bridge_sites(distance=distance)
            hollow = self._get_hollow_sites(distance=distance)
            return np.vstack((top, bridge, hollow))
        elif sitetype == 1:
            return self._get_top_sites(distance=distance)
        elif sitetype == 2:
            return self._get_bridge_sites(distance=distance)
        elif sitetype == 3:
            return self._get_hollow_sites(distance=distance)
    
        else:
            raise ValueError("sitetype not understood. Use -1, 1, 2 or 3")
    


    def get_cluster_descriptor(self, only_surface = False, bubblesize = 2.5, 
            rCut=5.0, NradBas=5, Lmax=5, crossOver=True, all_atomtypes=[]):
        """Takes a boolean only_surface as input (next to SOAP-specific arguments).
        Returns a 2D array with a soap feature vector on a row per atom 
        (per surface atom, if only_surface is set to True).
        bubblesize as an optional input defines how concave the surface can be.
        """

        alp, bet = soaplite.genBasis.getBasisFunc(rCut, NradBas) # input:(rCut, NradBas)
        soapmatrix = soaplite.get_soap_structure(self, alp, bet, rCut=rCut, NradBas=NradBas, Lmax=Lmax, crossOver=crossOver, all_atomtypes=all_atomtypes )
        self.cluster_descriptor = soapmatrix
        if only_surface == True:
            surfid = self.get_surface_atoms(bubblesize = bubblesize)
            soapmatrix= soapmatrix[surfid]
            return soapmatrix
        else:
            return self.cluster_descriptor


    def get_sites_descriptor(self, sitetype = -1, rCut=5.0, NradBas=5, Lmax=5, 
            crossOver=True, all_atomtypes=[]):
        """Takes an ASE atoms object and a 2D-array of site positions (next to SOAP-specific arguments).
        Returns a 2D array with a soap feature vector on a row per specified site 
        """
        if sitetype == -1:
            pos = np.vstack((self.site_positions[1], self.site_positions[2], self.site_positions[3]))
        else:
            pos = self.site_positions[sitetype]
        alp, bet = soaplite.genBasis.getBasisFunc(rCut, NradBas) # input:(rCut, NradBas)
        soapmatrix = soaplite.get_soap_locals(self, pos, alp, bet, rCut=rCut, NradBas=NradBas, Lmax=Lmax, crossOver=crossOver, all_atomtypes=all_atomtypes )
        if sitetype == -1:
            n_top_sites = self.site_positions[1].shape[0]
            n_bridge_sites = self.site_positions[2].shape[0]
            n_hollow_sites = self.site_positions[3].shape[0]
            self.sites_descriptor[1] = soapmatrix[:n_top_sites]
            self.sites_descriptor[2] = soapmatrix[n_top_sites: n_top_sites + n_bridge_sites]
            self.sites_descriptor[3] = soapmatrix[n_top_sites + n_bridge_sites :]
        else:
            self.sites_descriptor[sitetype] = soapmatrix
        return soapmatrix

    def get_unique_sites(self, sitetype = -1, threshold = 0.001, idx=[]):
        """Takes a 2D-array soapmatrix, a uniqueness-threshold and optionally a list of indices as input.
        Returns a list of indices.
        """
        if sitetype == -1:
            soapmatrix = np.vstack((self.sites_descriptor[1], self.sites_descriptor[2], self.sites_descriptor[3]))
        else:
            soapmatrix = self.sites_descriptor[sitetype]
        unique_lst = []
        if len(idx) == 0:
            print("no ids given")
        else:
            print("using ids",len(idx), len(soapmatrix) )
            assert len(idx) == len(soapmatrix), "give a list of indices of length %r" % len(soapmatrix) 

        dist_matrix = squareform(pdist(soapmatrix))
        K = soapmatrix.shape[0]
        n_features = soapmatrix.shape[1]
        fts_ids = np.zeros(K, dtype='int') -1
         
        #choosing random start point
        fts_ids[0] = np.random.choice(soapmatrix.shape[0])
         
        #finding next k-1
        min_dist = dist_matrix[fts_ids[0]]
        for i in range(1, K):
            if min_dist.max() / (1.0 * n_features) < threshold:
                # early stop based on threshold
                fts_ids = fts_ids[:i]
                break
            else:
                fts_ids[i] = np.argmax(min_dist)
                min_dist = np.minimum(min_dist, dist_matrix[fts_ids[i]])   
        return fts_ids


        idx = np.array(idx, dtype = int)
        if len(idx) != 0:
            unique_ids = idx[unique_lst]
        else:
            unique_ids = unique_lst

        return unique_ids


    def get_ranked_sites(self, sitetype = -1, K = None, idx=[], greedy = False, is_safe = False):
        """Takes a 2D-array soapmatrix, a uniqueness-threshold and optionally a list of indices as input.
        Returns a list of indices.
        """
        if sitetype == -1:
            soapmatrix = np.vstack((self.sites_descriptor[1], self.sites_descriptor[2], self.sites_descriptor[3]))
        else:
            soapmatrix = self.sites_descriptor[sitetype]
        ranked_lst = []
        if len(idx) == 0:
            print("no ids given")
        else:
            print("using ids",len(idx), len(soapmatrix) )
            #assert len(idx) == len(soapmatrix), "give a list of indices of length %r" % len(soapmatrix) 

        if K == None:
            # run over all datapoints
            K = soapmatrix.shape[0]

        print("K for fps", K, "matrix for FPS", soapmatrix.shape)    
        if is_safe:
            ranked_lst = _safe_fps(soapmatrix, K, greedy = greedy)
        else:
            ranked_lst = _fps(soapmatrix, K, greedy = greedy)



        idx = np.array(idx, dtype = int)
        if len(idx) != 0:
            ranked_ids = idx[ranked_lst]
        else:
            ranked_ids = ranked_lst

        assert len(ranked_ids) == len(set(ranked_ids)), "Error! Double counting in FPS! Use is_safe = True." 

        return ranked_ids



    def get_unique_surface_atoms(self, threshold = 0.001, idx=[]):
        """same as get_unique_sites()"""
        soapmatrix = self.cluster_descriptor
        unique_lst = []
        if len(idx) == 0:
            print("no ids given")
        else:
            print("using ids",len(idx), len(soapmatrix) )
            #assert len(idx) == len(soapmatrix), "give a list of indices of length %r" % len(soapmatrix) 

        dist_matrix = squareform(pdist(soapmatrix))
        K = soapmatrix.shape[0]
        n_features = soapmatrix.shape[1]
        fts_ids = np.zeros(K, dtype='int') -1
         
        #choosing random start point
        fts_ids[0] = np.random.choice(soapmatrix.shape[0])
         
        #finding next k-1
        min_dist = dist_matrix[fts_ids[0]]
        for i in range(1, K):
            if min_dist.max() / (1.0 * n_features) < threshold:
                # early stop based on threshold
                fts_ids = fts_ids[:i]
                break
            else:
                fts_ids[i] = np.argmax(min_dist)
                min_dist = np.minimum(min_dist, dist_matrix[fts_ids[i]])   
        return fts_ids


        idx = np.array(idx, dtype = int)
        if len(idx) != 0:
            unique_ids = idx[unique_lst]
        else:
            unique_ids = unique_lst

        return unique_ids



if __name__ == "__main__":
    print("main")
