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
    return support.ase_object + cluster.ase_object

def _get_zerosites_adsorption_vectors(structure, sitetype, site_ids):
    if sitetype == -1:
        top = structure.zero_site_positions[1]
        bridge = structure.zero_site_positions[2]
        hollow = structure.zero_site_positions[3]
        zero_sites = np.vstack((top, bridge, hollow))
        # normalized distance
        top = np.multiply(structure.adsorption_vectors[1], 1.0)
        bridge = np.multiply(structure.adsorption_vectors[2], 1.0)
        hollow = np.multiply(structure.adsorption_vectors[3], 1.0)
        adsorption_vectors = np.vstack((top, bridge, hollow))
        top = structure.site_surface_atom_ids[1]
        bridge = structure.site_surface_atom_ids[2]
        hollow = structure.site_surface_atom_ids[3]
        print(top.shape, bridge.shape, hollow.shape)
        # different shapes
        surface_atoms = np.empty(top.shape[0] + bridge.shape[0] + hollow.shape[0], dtype = object)
        surface_atoms[:top.shape[0]] = top.tolist()
        surface_atoms[top.shape[0]:top.shape[0] + bridge.shape[0]] = bridge.tolist()
        surface_atoms[top.shape[0] + bridge.shape[0]:top.shape[0] + bridge.shape[0] + hollow.shape[0]] = hollow.tolist()
        #surface_atoms = list(top)
        #surface_atoms.extend(list(bridge))
        #surface_atoms.extend(list(hollow))
        print(surface_atoms.shape)
    else:
        zero_sites = structure.zero_site_positions[sitetype]
        adsorption_vectors = np.multiply(structure.adsorption_vectors[sitetype], 1.0)
        surface_atoms = structure.site_surface_atom_ids[sitetype]

    if len(site_ids) != 0:
        return zero_sites[site_ids], adsorption_vectors[site_ids], surface_atoms[site_ids]
    else:
        return zero_sites, adsorption_vectors, surface_atoms



def all_cluster_support(cluster, support,  cluster_sitetype = -1, 
    support_sitetype = -1, bond_length = None, 
    is_support_vertical = True, 
    cluster_site_ids = [], support_site_ids = []):
    """
    all cluster-support orientations, organized in a matrix
    """
    cluster_sites = cluster.get_sites(cluster_sitetype)
    support_sites = support.get_sites(support_sitetype)

    cluster_zerosites, cluster_adsorption_vectors, cluster_surface_atoms = _get_zerosites_adsorption_vectors(cluster, cluster_sitetype, cluster_site_ids)
    support_zerosites, support_adsorption_vectors, support_surface_atoms = _get_zerosites_adsorption_vectors(support, support_sitetype, support_site_ids)

    orientations = np.empty((cluster_zerosites.shape[0], support_zerosites.shape[0]), dtype=object)
    
    #print(len(cluster_surface_atoms))
    #print(cluster_surface_atoms)
    #print(len(support_surface_atoms))
    #print(support_surface_atoms)
    
    print(cluster_zerosites.shape[0], support_zerosites.shape[0])

    # compute bond_length
    if not bond_length:
        bond_length_mat = _heuristic_bondlength_guess(cluster, support, cluster_surface_atoms, support_surface_atoms)

    else:
        bond_length_mat = np.zeros((cluster_zerosites.shape[0], support_zerosites.shape[0])) + bond_length
    print("bond length matrix", bond_length_mat.shape)
    #print(bond_length_mat)

    if is_support_vertical:
        # keep only z-component
        support_adsorption_vectors[:, 0] = 0.0
        support_adsorption_vectors[:, 1] = 0.0

    for i in range(cluster_zerosites.shape[0]):
        for j in range(support_zerosites.shape[0]):
            # translation
            #new_cluster = cluster.copy()
            new_cluster = cluster
            atoms = new_cluster.ase_object.copy()

            #print(bond_length_mat[i,j])
            cluster_site = cluster_zerosites[i] + cluster_adsorption_vectors[i] * bond_length_mat[i,j]
            #print(cluster_site, cluster_zerosites[i], cluster_adsorption_vectors[i], bond_length_mat[i,j])
            #new_cluster.translate(support_zerosites[j]  - cluster_site)
            atoms.translate(support_zerosites[j]  - cluster_site)

            #new_cluster.translate([0,0,10.0])
            #new_cluster.ase_object.translate([0, 0 , 10.0])

            # orientation of cluster onto support
            #new_cluster.rotate(cluster_adsorption_vectors[i], 
            #    v=-support_adsorption_vectors[j], 
            #    center=support_zerosites[j], 
            #    rotate_cell=False)
            atoms.rotate(cluster_adsorption_vectors[i], 
                v=-support_adsorption_vectors[j], 
                center=support_zerosites[j], 
                rotate_cell=False)
            from ase.visualize import view



            #print("ase object diff", (new_cluster.get_positions() - new_cluster.ase_object.get_positions()))
            #view(support.ase_object + atoms)
            orientations[i,j] = atoms
    return orientations
            

def unique_cluster_support(cluster, support,  
    cluster_sitetype = -1, support_sitetype = -1, 
    bond_length = None, is_support_vertical = True, threshold = 0.001):
    """
    unique cluster-support orientations
    """
    cluster.get_sites(cluster_sitetype)
    support.get_sites(support_sitetype)
    cluster.get_sites_descriptor(sitetype = cluster_sitetype)
    support.get_sites_descriptor(sitetype = support_sitetype)
    unique_cluster_lst = cluster.get_unique_sites(sitetype = cluster_sitetype, threshold = threshold, idx=[])
    unique_support_lst = support.get_unique_sites(sitetype = support_sitetype, threshold = threshold, idx=[])

    orientations = all_cluster_support(cluster, support,  cluster_sitetype = cluster_sitetype, 
        support_sitetype = support_sitetype, bond_length = bond_length, 
        is_support_vertical = is_support_vertical, 
        cluster_site_ids = unique_cluster_lst, support_site_ids = unique_support_lst)

    return orientations


def rank_cluster_support(cluster, support,  cluster_sitetype = -1, 
    support_sitetype = -1, bond_length = None, is_support_vertical = True, 
    cluster_site_ids = [], support_site_ids = []):
    """
    ranked cluster-support orientations based on local descriptor
    """
    cluster.get_sites(cluster_sitetype)
    support.get_sites(support_sitetype)
    cluster.get_sites_descriptor(sitetype = cluster_sitetype)
    support.get_sites_descriptor(sitetype = support_sitetype)

    ranked_cluster_lst = cluster.get_ranked_sites(sitetype = cluster_sitetype, K = None, idx=cluster_site_ids, greedy = False)
    ranked_support_lst = support.get_ranked_sites(sitetype = cluster_sitetype, K = None, idx=support_site_ids, greedy = False)

    orientations = all_cluster_support(cluster, support,  cluster_sitetype = cluster_sitetype, 
        support_sitetype = support_sitetype, bond_length = bond_length, 
        is_support_vertical = is_support_vertical, 
        cluster_site_ids = ranked_cluster_lst, support_site_ids = ranked_support_lst)

    return orientations

def rank_distance_cluster_support(cluster, support,  cluster_sitetype = -1, 
    support_sitetype = -1, bond_length = None, is_support_vertical = True,
    cluster_site_ids = [], support_site_ids = []):

    orientations, cs_distance = get_distance_cluster_support(cluster, support,  cluster_sitetype = cluster_sitetype, 
        support_sitetype = support_sitetype, bond_length = bond_length, is_support_vertical = is_support_vertical,
        cluster_site_ids = cluster_site_ids, support_site_ids = support_site_ids)

    orientations = orientations.flatten()

    cs_distance = cs_distance.flatten()

    sortids = np.argsort(cs_distance)

    orientations = orientations[sortids]

    return orientations


def get_distance_cluster_support(cluster, support,  cluster_sitetype = -1, 
    support_sitetype = -1, bond_length = None, is_support_vertical = True,
    cluster_site_ids = [], support_site_ids = []):
    """
    ranked cluster-support orientations based on the average distance 
    of the surface atoms of the cluster
    to the surface atoms of the support
    """
    orientations = all_cluster_support(cluster, support,  
        cluster_sitetype = cluster_sitetype, support_sitetype = support_sitetype, 
        bond_length = bond_length, is_support_vertical = is_support_vertical,
        cluster_site_ids = cluster_site_ids, support_site_ids = support_site_ids)

    n_cluster_sites, n_support_sites = orientations.shape

    cs_distance = np.zeros(orientations.shape)

    for i in range(n_cluster_sites):
        for j in range(n_support_sites):
            pos = orientations[i,j].get_positions()

            ase_dvec, ase_dmat = ase.geometry.get_distances(p1=pos, 
                p2=support.get_positions(), 
                cell=support.get_cell(), pbc=True)

            min_distances = ase_dmat.min(axis = 1)

            average_min_distance = min_distances.mean()

            # weighting
            weighted_distance = np.exp(-min_distances).mean()

            cs_distance[i,j] = average_min_distance

    # rank by cluster site or support site

    return orientations, cs_distance

def _heuristic_bondlength_guess(cluster, support, cluster_surface_atoms, support_surface_atoms):
    """
    bond_length guess based on atomic radii
    """
    bond_length_mat = np.zeros((len(cluster_surface_atoms), len(support_surface_atoms)))
    cluster_atomic_numbers = cluster.get_atomic_numbers()
    support_atomic_numbers = support.get_atomic_numbers()

    for i, c_site in enumerate(cluster_surface_atoms):
        for j, s_site in enumerate(support_surface_atoms):
            #print(i,j)
            if type(c_site) in (np.int32, int, np.int64):
                c_site = [c_site]
            if type(s_site) in (np.int32, int, np.int64):
                s_site = [s_site]
            c_radius = 0
            for atom_id in c_site:
                c_radius += covalent_radii[cluster_atomic_numbers[int(atom_id)]]
            c_radius /= float(len(c_site))

            s_radius = 0
            for atom_id in s_site:
                s_radius += covalent_radii[support_atomic_numbers[int(atom_id)]]
            s_radius /= float(len(s_site))

            bond_length_mat[i,j] = c_radius + s_radius
    return bond_length_mat


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
        min_distance = 2.0, max_distance = 1.9, 
        is_support_sites = True)

    from ase.visualize import view
    #view(support.ase_object + cluster.ase_object +  ase.Atoms('H' *len(sites), positions=sites))


    #orientations = all_cluster_support(cluster, support,  
    #    cluster_sitetype = 3, support_sitetype = 1, bond_length = 2.5, is_support_vertical = True)

    #orientations = rank_distance_cluster_support(cluster, support,  cluster_sitetype = 1, 
    #    support_sitetype = -1, bond_length = None, is_support_vertical = True,
    #    cluster_site_ids = [0,41], support_site_ids = [0,20,30,50,70,100])

    #orientations  = unique_cluster_support(cluster, support,  
    #    cluster_sitetype = -1, support_sitetype = -1, 
    #    bond_length = None, is_support_vertical = False, 
    #    threshold = 0.01)

    orientations  = rank_cluster_support(cluster, support,  
        cluster_sitetype = -1, support_sitetype = -1, 
        bond_length = None, is_support_vertical = False, 
        )



    #print(cs_distance.shape)
    #print("cs_distance")
    #print(cs_distance)

    # if you call get_distance_cluster_support to get cs_distance
    # you can filter by distance
    #orientations = orientations[cs_distance <= 6.55]


    print("final shape of orientations:", orientations.shape)

    from ase.io import Trajectory
    t1 = Trajectory('t1.traj', 'w')
    #configurations = []
    for orientation in orientations.flatten().tolist():
        #print(orientation, type(orientation))
        #configurations.extend(support.ase_object + orientation)
        t1.write(support.ase_object + orientation)

    t1.close()
    configurations = ase.io.read('t1.traj', index = ":")
    
    view(configurations)
