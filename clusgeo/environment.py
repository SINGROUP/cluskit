import numpy as np 
import ase, ase.io
import os, argparse
from ctypes import *
import soaplite
import pathlib
import clusgeo.surface

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


    idx = np.array(idx, dtype = int)
    if len(idx) == 0:
        unique_ids = idx[unique_lst]
    else:
        unique_ids = unique_lst

    return unique_ids




if __name__ == "__main__":

    atoms = ase.io.read("tests/au40cu40.xyz")
    
    soapmatrix = get_soap_cluster(atoms, only_surface=False, bubblesize=2.5, NradBas=10, Lmax =9)
    surfsoapmatrix = get_soap_cluster(atoms, only_surface=True, bubblesize=2.5, NradBas=10, Lmax =9)

    print("soap cluster shape", soapmatrix.shape)

    surfatoms = clusgeo.surface.get_surface_atoms(atoms)
    sitepositions = clusgeo.surface.get_top_sites(atoms, surfatoms)
    sitesoapmatrix = get_soap_sites(atoms, sitepositions, NradBas=10, Lmax =9)

    print("soap sites shape", sitesoapmatrix.shape)

    # unique sites testing

    unique_lst = get_unique_sites(soapmatrix, idx=surfatoms)




