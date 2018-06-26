import numpy as np 
import ase, ase.io
import os, argparse
from ctypes import *
import pathlib
import soaplite
import clusgeo.surface
from scipy.spatial.distance import squareform, pdist


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

def get_soap_cluster(obj, only_surface = False, bubblesize = 2.5, 
        rCut=5.0, NradBas=5, Lmax=5, crossOver=True, all_atomtypes=[]):
    """Takes an ASE atoms object and a boolean only_surface as input (next to SOAP-specific arguments).
    Returns a 2D array with a soap feature vector on a row per atom 
    (per surface atom, if only_surface is set to True).
    bubblesize as an optional input defines how concave the surface can be.
    """

    alp, bet = soaplite.genBasis.getBasisFunc(rCut, NradBas) # input:(rCut, NradBas)
    soapmatrix = soaplite.get_soap_structure(obj, alp, bet, rCut=rCut, NradBas=NradBas, Lmax=Lmax, crossOver=crossOver, all_atomtypes=all_atomtypes )

    if only_surface == True:
        surfid = clusgeo.surface.get_surface_atoms(obj, bubblesize = 2.5)
        soapmatrix= soapmatrix[surfid]
    return soapmatrix

def get_soap_sites(obj, pos, rCut=5.0, NradBas=5, Lmax=5, 
        crossOver=True, all_atomtypes=[]):
    """Takes an ASE atoms object and a 2D-array of site positions (next to SOAP-specific arguments).
    Returns a 2D array with a soap feature vector on a row per specified site 
    """
    alp, bet = soaplite.genBasis.getBasisFunc(rCut, NradBas) # input:(rCut, NradBas)
    soapmatrix = soaplite.get_soap_locals(obj, pos, alp, bet, rCut=rCut, NradBas=NradBas, Lmax=Lmax, crossOver=crossOver, all_atomtypes=all_atomtypes )
    return soapmatrix

def get_unique_sites(soapmatrix, threshold = 0.001, idx=[]):
    """Takes a 2D-array soapmatrix, a uniqueness-threshold and optionally a list of indices as input.
    Returns a list of indices.
    """
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


def rank_sites(soapmatrix, K = None, idx=[], greedy = False, is_safe = False):
    """Takes a 2D-array soapmatrix, a uniqueness-threshold and optionally a list of indices as input.
    Returns a list of indices.
    """
    ranked_lst = []
    if len(idx) == 0:
        print("no ids given")
    else:
        print("using ids",len(idx), len(soapmatrix) )
        assert len(idx) == len(soapmatrix), "give a list of indices of length %r" % len(soapmatrix) 

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
