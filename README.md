# ClusGeo4

[![Build Status](https://travis-ci.org/SINGROUP/ClusGeo3.0.svg?branch=master)](https://travis-ci.org/SINGROUP/ClusGeo3.0)
[![Coverage Status](https://coveralls.io/repos/github/SINGROUP/ClusGeo3.0/badge.svg?branch=master)](https://coveralls.io/github/SINGROUP/ClusGeo3.0?branch=master)


ClusGeo is a python package dedicated to surface science on nanoclusters. ClusGeo currently focuses on the population of given nanoclusters with adsorbates. Furthermore, at a later stage it will also feature the generation of a multitude of different types of nanoclusters.

Sites on the nanoclusters are classified as top, bridge and hollow. One can automatically detect surface atoms as well as those sites on arbitrary nanoclusters. Using a structural descriptor those surface atoms or sites can be compared with each other with respect to their (dis)similarity.

ClusGeo is especially designed for creating many samples of adsorbates on a variety of nanoclusters without the need of visually checking the structures. It is complementary to the python package for surface slabs KatKit https://github.com/SUNCAT-Center/CatKit

OLD DOCUMENTATION!
breaking changes since then!
will be updated in the near future...

Currently ClusGeo has limited capabilities which will be enhanced in the near future:

## clusgeo.surface.get_surface_atoms(...)

Gets surface atom indices

## clusgeo.surface.get_non_surf_atoms(...)

Gets nonsurface atom indices

## clusgeo.surface.get_top_sites(...)  as well as _edge_ and _hollow_

Gets all possible sites of class top,bridge or hollow.

## clusgeo.surface.x2_to_x(...)

Eliminates a single atom adsorbate as soon as it is too close to another.

## clusgeo.surface.write_all_sites(...)

Utility function to write xyz files in a directory structure split by top, bridge and hollow.

## clusgeo.environment.get_soap_cluster(...)

Get a matrix of size NxM where N is the number of atoms in the cluster and M is the number of SOAP features

## clusgeo.environment.get_soap_sites(...)

Given a cluster and position of adsorbates, get a matrix of size NxM where N is the number of adsorbates on the cluster and M is the number of SOAP features

## clusgeo.environment.get_unique_sites(...)

Given a certain threshold, eliminate too similar sites in SOAP feature space.

## clusgeo.environment.rank_sites(...)

Similar to .get_unique_sites(...) but instead of giving a threshold, all sites can be ranked or optionally only the top K can be determined. Ranking is done using farthest point sampling.


# Example

example coming soon

# Dependencies and installation

ClusGeo depends heavily on ASE, numpy, scipy and to a smaller degree on soaplite. soaplite is an implementation of the SOAP descriptor. This dependency will later be changed to the descriptor package dscribe

The newest version is available from github.
Clone the repository. Inside it run "python3 setup.py install"

An older version is available through "pip install clusgeo"


# Tests

In the folder tests the script test.py runs simple tasks on a sample cluster such as spotting surface atoms, finding top, bridge and hollow sites, ranking sites and determining unique sites.

We are planning to include regtests at a later stage

# Contributors

Eiaki Morooka
Marc Jäger
