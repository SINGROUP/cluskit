# functionalities involving more than one Cluster or Support
# might be turned into a System class in the future
import numpy as np 
from ase.data import covalent_radii
import ase, ase.io
import os, argparse, glob
from scipy.spatial.distance import cdist, squareform, pdist
import dscribe, dscribe.descriptors

def exclude_too_close_sites(cluster, support, cluster_sitetype = -1, min_distance = 2.0):
    """
    exclude sites too close to support

    either give cluster object with already computed sites
    or the pure sites

    cluster and support already need to be properly rotated and translated
    """
    cluster_sites = cluster.get_sites(cluster_sitetype)

    ase_dvec, ase_dmat = ase.geometry.get_distances(p1=cluster_sites, 
        p2=support.get_positions(), 
        cell=support.get_cell(), pbc=True)
    min_distances = ase_dmat.min(axis = 1)

    mask =  min_distances > min_distance
    return cluster_sites[mask]


def interface_sites(cluster, support, sitetype = -1, 
    min_distance = 2.0, max_distance = 1.0, is_support_sites = False):
    """
    get sites of the cluster which are only distance away
    from the support.

    if is_support_sites is set to True,
    instead sites of the support which are only distance away 
    from the cluster are returned
    """    

    if is_support_sites == True:
        sites = support.get_sites(sitetype)
        pos = cluster.get_positions()
    else:
        sites = cluster.get_sites(sitetype)
        pos = support.get_positions()

    ase_dvec, ase_dmat = ase.geometry.get_distances(p1=sites, 
        p2=pos, 
        cell=support.get_cell(), pbc=True)
    min_distances = ase_dmat.min(axis = 1)
    mask =  (min_distances < min_distance) & (min_distances > max_distance)
    return sites[mask]


def attach_cluster(cluster, support):
    """
    simple function to return
    object of combined cluster and support.
    cluster and support already need to be properly rotated and translated
    """
    return cluster + support


def unique_cluster_support(cluster, support,  cluster_sitetype = -1, support_sitetype = -1, bond_length = None, threshold = 0.001):
    """
    unique cluster-support orientations
    """

    return


def rank_cluster_support(cluster, support,  cluster_sitetype = -1, support_sitetype = -1, bond_length = None):
    """
    ranked cluster-support orientations based on local descriptor
    """
    return


def rank_distance_cluster_support(cluster, support,  cluster_sitetype = -1, support_sitetype = -1, bond_length = None):
    """
    ranked cluster-support orientations based on the average distance 
    of the surface atoms of the cluster
    to the surface atoms of the support
    """

    return

def _heuristic_bondlength_guess(type1, type2):
    """
    bond_length guess based on atomic radii
    """
    bond_length = covalent_radii(type1) + covalent_radii(type2)
    return bond_length


# TODO: add arguments cluster_sitetype, support_sitetype
# to define connection


if __name__ == '__main__':
    from ase.build import fcc111
    slab = fcc111('Al', size=(3,3,3), vacuum=10.0)
    atoms = slab
    from cluskit import Support
    support = Support(slab)
    from ase.cluster.icosahedron import Icosahedron
    atoms = Icosahedron('Cu', noshells=3)

    from cluskit import Cluster
    cluster = Cluster(atoms)

    cluster_sites = exclude_too_close_sites(cluster, support, cluster_sitetype = 1, min_distance = 3.0)
    print(cluster_sites)
    sites = interface_sites(cluster, support, sitetype = -1, 
        min_distance = 2.0, max_distance = 1.9, is_support_sites = True)

    from ase.visualize import view
    view(support.ase_object + cluster.ase_object +  ase.Atoms('H' *len(sites), positions=sites))