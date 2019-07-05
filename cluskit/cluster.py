import numpy as np 
import ase, ase.io
import os, argparse, glob
from ctypes import *
from scipy.spatial.distance import cdist, squareform, pdist
from cluskit.delaunay import delaunator
import cluskit.utils
import dscribe, dscribe.descriptors

def _fps(pts, K, greedy=False):
    """Helper function farthest point sampling to
    rank in an array of points (pts) those K points 
    furthest away from each other.

    Args:
        pts (2D ndarray) :  points in n-dimensional space. Can be e.g. 
                            real or feature space
        K (int)          :  Early stop criterion, can be as large as pts
                            itself. The search stops at the K-th furthest 
                            point
        greedy (bool)   :   algorithm switch. The greedy algorithm is a bit
                            faster but is less stable, in the sense that
                            two identical points could be double-counted

    Returns:
        1D ndarray :    ordered indices of length K of the original points
    """
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

def _rank_fps(pts, K, greedy=False):
    """Helper function farthest point sampling to
    rank in an array of points (pts) those K points 
    furthest away from each other. 
    If K is None, all points are ranked

    Args:
        pts (2D ndarray) :  points in n-dimensional space. Can be e.g. 
                            real or feature space
        K (int)          :  Early stop criterion, If set to None, K will
                            be as large as pts itself. 
                            The search stops at the K-th furthest point. 
        greedy (bool)   :   algorithm switch. The greedy algorithm is a bit
                            faster but is less stable, in the sense that
                            two identical points could be double-counted

    Returns:
        1D ndarray :    ordered indices of length K of the original points
    """
    if K == None:
        # run over all datapoints
        K = pts.shape[0]
    ranked_lst = _fps(pts, K, greedy = greedy)

    return ranked_lst

def _closest_node(node, nodes):
    """Helper function to find the closest node in an array
    of nodes

    Args:
        node (1D ndarray) : node, e.g. point in real space
        nodes (2D ndarray) :    nodes to compare against, e.g. list
                                of nodes in real space. The last dimension
                                needs to match with node

    Returns:
        tuple : index of closest node from nodes,
                their distance
    """
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2), np.min(dist_2)

def _unique_selection(descmatrix, threshold):
    """Helper function to find the unique instances
    of datapoints with respect to a descriptor matrix.
    Two datapoints are defined as unique if their distance
    d satisfies:

    d / (n_features) > threshold

    while n_features is the number of features in the 
    descriptor matrix and threshold an empirical parameter
    to tune the uniqueness.

    Farthest point sampling is used as the algorithm to
    find the "most unique" points first.

    Args:
        descmatrix (2D ndarray) :   descriptor matrix of a set of datapoints
        threshold (float) :     empirical parameter to define uniqueness

    Returns:
        1D ndarray :    ordered indices (by uniqueness) of variable 
                        size of the datapoints given by descmatrix
    """
    dist_matrix = squareform(pdist(descmatrix))
    K = descmatrix.shape[0]
    n_features = descmatrix.shape[1]
    fts_ids = np.zeros(K, dtype='int') -1
     
    #choosing random start point
    fts_ids[0] = np.random.choice(descmatrix.shape[0])
     
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

def _constrain_descmatrix_to_selected_ids(descmatrix, idx):
    """Helper function to reduce the descriptor matrix to
    specific datapoints given by indices.

    Args:
        descmatrix (2D ndarray) :   descriptor matrix of a set of datapoints
        idx (1D ndarray) :    indices to slice the matrix with

    Returns:
        2D ndarray :    descriptor matrix, reduced and ordered by idx
    """
    idx = np.array(idx, dtype = int)
    if len(idx) != 0:
        return descmatrix[idx]
    else:
        return descmatrix

def _translate_to_selected_ids(unique_ids, idx):
    """Helper function to translate the unique ids to
    selected ids
    """
    idx = np.array(idx, dtype = int)
    if len(idx) != 0:
        return idx[unique_ids]
    else:
        return unique_ids

def _average_minimum_distance(positions):
    """Helper function to determine the average minimum distance
    in a set of points.

    This function is important to ensure a robust guess of the 
    max_bondlength parameter for the delaunay algorithm.

    Args:
        positions (2D ndarray) :    points in n-dimensional space. Can be e.g. 
                                    real or feature space

    Returns:
        float : average minimum distance between points
    """
    dist_matrix = squareform(pdist(positions))
    ma = np.ma.masked_equal(dist_matrix, 0.0, copy=False)
    minimums = np.min(ma, axis = 0)
    avemin = minimums.mean()
    return avemin

class Cluster(ase.Atoms):
    """ A child class of the ase.Atoms object. 
    It is a nanocluster class with additional methods and attributes, 
    built for the detection and classification of surfaces and adsorption sites. 

    Args:

    Returns:
        cluskit.Cluster : cluster object
    """

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
        """Upon initialization, the surface is defined using the
        delaunay algorithm. Other attributes get defaults such
        as self.descriptor_setup = SOAP, they can be overwritten
        afterwards.
        """
        self.ase_object = super(Cluster, self).__init__(symbols=symbols,
                    positions=positions, numbers=numbers,
                    tags=tags, momenta=momenta, masses=masses,
                    magmoms=magmoms, charges=charges,
                    scaled_positions=scaled_positions,
                    cell=cell, pbc=pbc, celldisp=celldisp,
                    constraint=constraint,
                    calculator=calculator,
                    info=info)
        self.ase_object = ase.Atoms(symbols=symbols,
                    positions=positions, numbers=numbers,
                    tags=tags, momenta=momenta, masses=masses,
                    magmoms=magmoms, charges=charges,
                    scaled_positions=scaled_positions,
                    cell=cell, pbc=pbc, celldisp=celldisp,
                    constraint=constraint,
                    calculator=calculator,
                    info=info)
        
        self.site_surface_atom_ids = {}
        self.zero_site_positions = {}
        self.adsorption_vectors = {}

        # empiric multiplier, used for determining surface atoms
        self.max_bondlength = _average_minimum_distance(self.positions) * 1.7

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
        self.surface_atoms = self.arrays['surface']

        # setting standard descriptor
        atomic_numbers = sorted(list(set(self.get_atomic_numbers())))
        self.descriptor_setup = dscribe.descriptors.SOAP(
            species=atomic_numbers,
            periodic=False,
            rcut=5.0,
            nmax=8,
            lmax=6,
            sparse=False,
            )


    def get_surface_atoms(self, mask=False):
        """Determines the surface atoms of the nanocluster using delaunay
        triangulation.
        A maximum bondlength determining how concave a surface can be can be 
        adjusted in the attribute self.max_bondlength
        (the default usually works well).

        The attributes
        -self.zero_site_positions
        -self.adsorption_vectors
        are populated

        Args: 
            mask (bool) :   If set to True, a mask array will be returned.
                            If False, an array of indices of surface atoms 
                            is returned

        Returns:
            1D ndarray : indices of surface atoms
        """
        if self.has('surface'):
            if mask:
                surface_mask = np.zeros(len(self), dtype='bool')
                surface_mask[self.arrays['surface']] = True
                return surface_mask
            else:
                return self.arrays['surface']
 
        pos = self.get_positions()

        # delaunator not only gets surface atoms but also adsorption site locations
        summary_dict = delaunator(pos, self.max_bondlength)

        ids_surface_atoms = summary_dict["ids_surface_atoms"]
        
        # store in attributes
        self.arrays['surface'] = ids_surface_atoms

        self.site_surface_atom_ids[1] = ids_surface_atoms
        self.zero_site_positions[1] = summary_dict["positions_surface_atoms"] 
        self.adsorption_vectors[1] = summary_dict["normals_surface_atoms"] 
        
        self.site_surface_atom_ids[2] = summary_dict["ids_surface_edges"] 
        self.zero_site_positions[2] = summary_dict["centers_surface_edges"]
        self.adsorption_vectors[2] = summary_dict["normals_surface_edges"] 
        
        self.site_surface_atom_ids[3] = summary_dict["ids_surface_triangles"] 
        self.zero_site_positions[3] = summary_dict["centers_surface_triangles"] 
        self.adsorption_vectors[3] = summary_dict["normals_surface_triangles"]

        if mask:
            surface_mask = np.zeros(len(self), dtype='bool')
            surface_mask[ids_surface_atoms] = True
            return surface_mask
        else:
            return ids_surface_atoms

    def get_nonsurface_atoms(self):
        """ Determines the core / non-surface atoms of the nanocluster. 
        A maximum bondlength determining how concave a surface can be can be adjusted
        in the attribute self.max_bondlength    
        (the default usually works well).
    
        Args: 

        Returns:
            1D ndarray : indices of core / non-surface atoms
        """
        surface = self.get_surface_atoms(mask=True)
        return np.nonzero(np.logical_not(surface))[0]

    def _compute_adsorbate_positions(self, sitetype, distance=1.8):
        """Helper function to determine the location of adsorbates on a given
        sitetype.
        The adsorbates will be placed away from the zerosite, in the direction 
        of the adsorption vectors.

        The zerosite is the surface atom in the case of a top site,
        the middle between the two atoms of a bridge site,
        and the geometrical center of the face of a hollow site

        Args:
        sitetype (int)      :   1 : "top", 2 : "bridge", 3 : "hollow" site
        distance (float)  :     distance from zerosite to adsorbate        

        Returns:
            2D ndarray :    site positions (of type sitetype) with the defined 
                            distance from the adsorbing zerosite.
        """
        self.get_surface_atoms()

        zero_site = self.zero_site_positions[sitetype]
        v = np.multiply(self.adsorption_vectors[sitetype], distance)

        site_positions = zero_site + v
        self.site_positions[sitetype] = site_positions
        return self.site_positions[sitetype]    


    def get_sites(self, sitetype=-1,  distance= 1.8):
        """This method gets top, bridge or hollow adsorption sites on the nanocluster.
        
        The adsorbates will be placed away from the zerosite, in the direction 
        of the adsorption vectors.

        The zerosite is the surface atom in the case of a top site,
        the middle between the two atoms of a bridge site,
        and the geometrical center of the face of a hollow site

        If sitetype = -1, top, bridge and hollow sites are concatenated in that order.

        Args:
        sitetype (int)      :   1 : "top", 2 : "bridge", 3 : "hollow" site
                                -1 : all of the above
        distance (float)  :     distance from zerosite to adsorbate        

        Returns:
            2D ndarray :    site positions (of type sitetype) with the defined 
                            distance from the adsorbing zerosite.
        """
        if sitetype == -1:
            top = self._compute_adsorbate_positions(sitetype = 1, distance=distance)
            bridge = self._compute_adsorbate_positions(sitetype = 2, distance=distance)
            hollow = self._compute_adsorbate_positions(sitetype = 3, distance=distance)
            return np.vstack((top, bridge, hollow))
        elif sitetype == 1:
            return self._compute_adsorbate_positions(sitetype = 1, distance=distance)
        elif sitetype == 2:
            return self._compute_adsorbate_positions(sitetype = 2, distance=distance)
        elif sitetype == 3:
            return self._compute_adsorbate_positions(sitetype = 3, distance=distance)
        else:
            raise ValueError("sitetype not understood. Use -1, 1, 2 or 3")


    def customize_sites(self, surface_atom_ids, 
            sitetype = -1, is_exclusive = False):
        """
        Takes a custom list of surface atom ids (the atoms have to be
        on the surface). Optionally takes sitetype and is_exclusive as input.

        In order to influence the distance to the zerosite, run self.get_sites()
        with the desired distance

        Args:
            surface_atom_ids (1D ndarray) : a subset of surface atom indices
            sitetype (int)      :   1 : "top", 2 : "bridge", 3 : "hollow" site
                                    -1 : all of the above
            is_exclusive (bool) :   If set to True, returns only sites consisting 
                                    of exclusively the given surface atoms. 
                                    Otherwise, by default, returns sites 
                                    with at least one of them being in 
                                    surface_atom_ids
        Returns:
            list or dict :      a reduced set of adsorption sites, previously
                                defined through the self.get_sites() method
        """        
        # Check if all custom surface_atom_ids are actually on the surface
        if np.all(np.isin(surface_atom_ids, self.surface_atoms)):
            pass
        else:
            raise Exception("surface_atom_ids should be in surface_atoms")

        if sitetype == -1:
            sitetypes = [1,2,3]
        else:
            sitetypes = [sitetype]

        custom_sites = {}
        for tp in sitetypes:

            site_surface_atom_ids  = self.site_surface_atom_ids[tp]
            if tp == 1:
                site_surface_atom_ids = site_surface_atom_ids.reshape((-1,1))

            is_in  = np.isin(site_surface_atom_ids, surface_atom_ids)
            if is_exclusive:
                mask = np.all(is_in, axis = -1)
            else:
                mask = np.any(is_in, axis = -1)

            custom_sites[tp] = np.flatnonzero(mask)

        if sitetype == -1:
            return custom_sites
        else:
            return custom_sites[sitetype]

    def find_closest_site(self, position):
        """Takes a point in space as input.
        Returns the sitetype and the index of the closest site 
        of the cluster. (Had to be stored previously)

        In order to influence the distance to the zerosite, call self.get_sites()
        with the desired distance

        Args:
            position (1D ndarray) : point in real space

        Returns:
            tuple :     sitetype,
                        index of closest site
        """
        idx1, dist1 = _closest_node(position, self.site_positions[1])
        idx2, dist2 = _closest_node(position, self.site_positions[2])
        idx3, dist3 = _closest_node(position, self.site_positions[3])

        sitetype = np.argmin([dist1, dist2, dist3]) + 1
        closest_site = [idx1, idx2, idx3][sitetype - 1]

        return sitetype, closest_site

    def get_ase_atomic_adsorbates(self, sitetype = -1, distance = 1.8, atomtype = "X"):
        """Only for single-atom adsorbate.

        Args:
            sitetype (int)      :   1 : "top", 2 : "bridge", 3 : "hollow" site
                                    -1 : all of the above
            distance (float)  :     distance from zerosite to adsorbate
            atomtype (str) :    atomic symbol of the atomic adsorbate

        Returns:
            list : ase.Atoms objects at the adsorption site positions
        """
        positions = self.get_sites(sitetype = sitetype, distance = distance)
        adsorbates = []
        for position in positions:
            atom = ase.Atoms(symbols=atomtype, positions=position.reshape((1,3)))
            adsorbates.append(atom)
        return adsorbates

    def place_adsorbates(self, molecule, sitetype = -1, remove_x = True):
        """
        This method places an adsorbate (ase object containing X dummy atom) onto top, 
        bridge and/or hollow adsorption sites on the nanocluster.

        Returns a 2D-array of top, bridge, and/or hollow site positions with the 
        defined distance from the adsorbing surface atom(s).
        If sitetype = -1, top, bridge and hollow sites are concatenated in that order.

        Args:
            molecule (ase.Atoms) :  this object needs to contain one dummy/'X' atom
                                    which anchors to the zerosite. The adsorption 
                                    vector of that site is aligned with the vector
                                    from 'X' to its closest atom in the molecule
            sitetype (int)      :   1 : "top", 2 : "bridge", 3 : "hollow" site
                                    -1 : all of the above
            remove_x (bool) :   should the dummy atom be stripped from the 
                                atoms objects

        Returns:
            list :  ase.Atoms objects of molecular adsorbates at the 
                    adsorption site positions
        """
        if sitetype == -1:
            zero_sites = np.vstack((self.zero_site_positions[1], 
                self.zero_site_positions[2], self.zero_site_positions[3]))
            adsorption_vectors = np.vstack((self.adsorption_vectors[1], 
                self.adsorption_vectors[2], self.adsorption_vectors[3]))
        elif sitetype in [1,2,3]:
            zero_sites = self.zero_site_positions[sitetype]
            adsorption_vectors = self.adsorption_vectors[sitetype]
        else:
            raise ValueError("sitetype not understood. Use -1, 1, 2 or 3")

        adsorbate_lst = []
        for zero_site, adsorption_vector in zip(zero_sites, adsorption_vectors):
            adsorbate = cluskit.utils.place_molecule_on_site(molecule, zero_site, 
                adsorption_vector, remove_x = remove_x)
            adsorbate_lst.append(adsorbate)

        return adsorbate_lst


    def get_cluster_descriptor(self, only_surface = False):
        """Takes a boolean only_surface as input, 
        and optionally a maximum bondlength which defines how concave the surface can be.
        Returns a 2D array with a descriptor (default is SOAP) feature vector on a 
        row per atom 
        (per surface atom, if only_surface is set to True).

        Args:
            only_surface (bool) :   if set to True, only the descriptor features of the
                                    surface atoms are returned

        Returns:
            2D ndarray :    descriptor matrix of all atoms (or only surface atoms)
                            of the nanocluster
        """
        desc = self.descriptor_setup
        descmatrix = desc.create(self.ase_object)
        self.cluster_descriptor = descmatrix
        if only_surface == True:
            surfid = self.get_surface_atoms()
            descmatrix = descmatrix[surfid]
            return descmatrix
        else:
            return self.cluster_descriptor


    def get_sites_descriptor(self, sitetype = -1):
        """Gets the descriptor (default is SOAP) feature 
        vector on a row per site (not zerosite!).
        If sitetype = -1, top, bridge and hollow sites are concatenated in that order.

        Args:
            sitetype (int)      :   1 : "top", 2 : "bridge", 3 : "hollow" site
                                    -1 : all of the above
        Returns:
            2D ndarray :    descriptor matrix of all sites of the given type
        """
        if sitetype == -1:
            pos = np.vstack((self.site_positions[1], 
                self.site_positions[2], self.site_positions[3]))
        else:
            pos = self.site_positions[sitetype]
        desc = self.descriptor_setup
        descmatrix = desc.create(self.ase_object, positions = pos.tolist())
        if sitetype == -1:
            n_top_sites = self.site_positions[1].shape[0]
            n_bridge_sites = self.site_positions[2].shape[0]
            n_hollow_sites = self.site_positions[3].shape[0]
            self.sites_descriptor[1] = descmatrix[:n_top_sites]
            self.sites_descriptor[2] = descmatrix[n_top_sites: n_top_sites + n_bridge_sites]
            self.sites_descriptor[3] = descmatrix[n_top_sites + n_bridge_sites :]
        else:
            self.sites_descriptor[sitetype] = descmatrix
        return descmatrix

    def get_unique_sites(self, sitetype = -1, threshold = 0.001, idx=[]):
        """Takes firstly a sitetype as input. Valid are 1 (top), 2 (bridge) and 3 (hollow) 
        next to the default (default = -1 means top, bridge and hollow sites). Secondly, 
        a threshold (float) of uniqueness and optionally a list of indices are taken as input.
        Returns a list of indices of the unique sites.
        
        In order to influence the distance to the zerosite, call self.get_sites()
        with the desired distance

        Args:
            sitetype (int)      :   1 : "top", 2 : "bridge", 3 : "hollow" site
                                    -1 : all of the above
            threshold (float) :     empirical parameter to define uniqueness
            idx (1D ndarray) :  custom indices to translate the unique indices to

        Returns:
            1D ndarray : indices of the unique sites. The actual sites can be retrieved
                         from the self.get_sites() method or the self.site_positions
                         attribute
        """
        if sitetype == -1:
            descmatrix = np.vstack((self.sites_descriptor[1], 
                self.sites_descriptor[2], self.sites_descriptor[3]))
        else:
            descmatrix = self.sites_descriptor[sitetype]

        descmatrix = _constrain_descmatrix_to_selected_ids(descmatrix, idx)

        unique_ids = _unique_selection(descmatrix = descmatrix, threshold = threshold)

        unique_ids = _translate_to_selected_ids(unique_ids, idx)

        return unique_ids


    def get_ranked_sites(self, sitetype = -1, K = None, idx=[], greedy = False):
        """Takes firstly a sitetype as input. Valid are 1 (top), 2 (bridge) and 3 (hollow) 
        next to the default (default = -1 means top, bridge and hollow sites). Secondly, 
        the sites are ranked up to K (int). If K = None, all sites are ranked. 
        Thirdly, it optionally a list of indices as input, as well as the boolean argument 
        greedy  which refers to the farthest point sampling algorithm used. 
        Returns a list of indices of the ranked sites (of length K).
        In order to influence the distance to the zerosite, call self.get_sites()
        with the desired distance

        Args:
            sitetype (int)      :   1 : "top", 2 : "bridge", 3 : "hollow" site
                                    -1 : all of the above
            K (int)          :  Early stop criterion, If set to None, K will
                                be as large as the number of sites itself. 
                                The indices are pruned to the K-th most 
                                dissimilar sites.            
            idx (1D ndarray) :  custom indices to translate the ranked indices to

        Returns:
            1D ndarray : indices of the ranked sites. The actual sites can be retrieved
                         from the self.get_sites() method or the self.site_positions
                         attribute
        """
        if sitetype == -1:
            descmatrix = np.vstack((self.sites_descriptor[1], 
                self.sites_descriptor[2], self.sites_descriptor[3]))
        else:
            descmatrix = self.sites_descriptor[sitetype]

        descmatrix = _constrain_descmatrix_to_selected_ids(descmatrix, idx)

        ranked_ids = _rank_fps(descmatrix, K, greedy=False)

        ranked_ids = _translate_to_selected_ids(ranked_ids, idx)

        return ranked_ids


    def get_unique_cluster_atoms(self, threshold = 0.001, idx=[]):
        """Method similar to .get_unique_sites(). Takes a threshold (float) 
        of uniqueness and optionally a list of indices as input.
        Returns a list of indices of the unique cluster atoms.

        Args:
            threshold (float) :     empirical parameter to define uniqueness
            idx (1D ndarray) :  custom indices to translate the unique indices to

        Returns:
            1D ndarray : indices of the unique cluster atoms.
        """
        descmatrix = self.cluster_descriptor
    
        descmatrix = _constrain_descmatrix_to_selected_ids(descmatrix, idx)

        unique_ids = _unique_selection(descmatrix = descmatrix, threshold = threshold)

        unique_ids = _translate_to_selected_ids(unique_ids, idx)

        return unique_ids
