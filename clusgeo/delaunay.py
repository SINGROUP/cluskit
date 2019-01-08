import numpy
from itertools import combinations


## \brief Computes one of the centers of a triangle. (Do not remember which one)
##
## \param coords: a 3x3 matrix with the xyz coords of the vertexes on each row.
## \return The center point as numpy array.
def FaceCenter(coords):

    d = numpy.zeros((3))
    d[0] = numpy.linalg.norm(coords[1]-coords[2])
    d[1] = numpy.linalg.norm(coords[2]-coords[0])
    d[2] = numpy.linalg.norm(coords[0]-coords[1])
    p = d[0] + d[1] + d[2]
    d /= p
    c = numpy.zeros((3))
    for k in range(3): c += coords[k] * d[k] 

    return c

## \brief Computes the normal vector of a triangle.
##
## \param coords: numpy 3x3 matrix with the triangle verteces xyz components on the rows.
## \return Normal vector as numpy array.
def FaceNormal(coords, normalise=True):

    r0 = coords[0]
    v1 = coords[1] - r0
    v2 = coords[2] - r0
    normal = numpy.cross(v1, v2)
    if normalise: normal /= numpy.linalg.norm(normal)
    
    return normal


def FindSameTri(tri, faces):

    found = False
    
    for i in range(len(faces)):
        f = faces[i][0]
        if tri[0] == f[0] and tri[1] == f[1] and tri[2] == f[2]:
            found = True
            faces.pop(i)
            break

    return found, faces

## \brief This is the shit!
##
## \param xyz: numpy matrix N (number of atoms) rows, 3 columns (x,y,z position).
## \param rcut: max distance between atoms that can be considered a side of a tetrahedron.
##
## 
def delaunator(xyz, rcut):

    natm = xyz.shape[0]
    r2 = rcut * rcut

    tetras = [[0,0,0,0]]
    centers = [[0,0,0,0]]
    
    faces = numpy.asarray([[0,0,0]])
    normals = numpy.asarray([[0,0,0]])
    tris = []
    incenters = [[0,0,0]]

    allinfo = []
    
    for i in range(natm):
        for j in range(i+1, natm):
            d = xyz[i]-xyz[j]; d = numpy.dot(d,d)
            if d > r2: continue

            for k in range(j+1, natm):
                d = xyz[i]-xyz[k]; d = numpy.dot(d,d)
                if d > r2: continue
                d = xyz[j]-xyz[k]; d = numpy.dot(d,d)
                if d > r2: continue

                for l in range(k+1, natm):
                    d = xyz[i]-xyz[l]; d = numpy.dot(d,d);
                    if d > r2: continue
                    d = xyz[j]-xyz[l]; d = numpy.dot(d,d);
                    if d > r2: continue
                    d = xyz[k]-xyz[l]; d = numpy.dot(d,d);
                    if d > r2: continue

                    # code here means all distances were ok!
                    t = [i,j,k,l]
                    #tetra = tetra + [t]

                    # compute circumsphere center and radius
                    ri = xyz[t[1:]]
                    r0 = xyz[t[0]]
                    d2 = numpy.zeros((3))
                    
                    for c in range(3):
                        ri[c] -= r0
                        d2[c] = numpy.dot(ri[c], ri[c])
                    
                    c = numpy.linalg.solve(2*ri, d2)
                    r = numpy.linalg.norm(c)
                    c += r0
                    
                    # check if this sphere contains other vertexes
                    othersInside = False
                    d2 = r * r
                    for o in range(natm):
                        if o in t: continue
                        d = xyz[o] - c; d = numpy.dot(d,d)
                        if d < d2:
                            othersInside = True
                            break
                    # not a good delaunay tetrahedron!
                    if othersInside: continue

                    # the tetra is a good one!
                    #tetras = tetras + [t]
                    #centers = centers + [[c[0], c[1], c[2], r]]
                    
                    # check if the tris faces are already in the list
                    # if a tris has a duplicate, then it is not a surface tris

                    # these are sorted tris indexes
                    tfaces = numpy.asarray([[i,j,k],[i,j,l],[i,k,l],[j,k,l]])

                    # make sure each tris is in the right order, so the normal points out
                    tetraCoords = xyz[t]
                    # for each face, compute the area
                    Ai = numpy.zeros((4))
                    for f in range(4):
                        coords = xyz[tfaces[f]]
                        Ai[f] = 0.5 * numpy.linalg.norm(numpy.cross(coords[1]-coords[0], coords[2]-coords[0]))

                    # compute the incenter of the tetrahedron
                    incenter = numpy.zeros((3))
                    for cc in range(4): incenter += Ai[cc] * tetraCoords[cc]
                    incenter /= numpy.sum(Ai)
                    incenters = incenters + [incenter]

                    tetraNormals = numpy.zeros((4,3))
                    
                    for f in range(4):

                        tri = tfaces[f]
                        coords = xyz[tri]

                        # compute normal
                        triNormal = FaceNormal(coords, False)
                        triCenter = FaceCenter(coords)

                        #tetraNormals[f] = triCenter - incenter
                        
                        if numpy.dot(triCenter - incenter, triNormal) < 0:
                            newtri = numpy.flip(tri, axis=0)
                            triNormal *= -1
                        else:
                            newtri = tri
                        # newtri is the verts indexes in the correct culling order!
                        # tri indexes are always sorted
                        
                        # now check if a face with the same tri indexes already exists
                        # if so, remove that face, and do not add this one!
                        found, newtris = FindSameTri(tri, tris)
                        tris = newtris
                        if not found:
                            tris = tris + [[tri,triNormal,triCenter,newtri, triNormal/numpy.linalg.norm(triNormal)]]
                        
                        
    # loop done... we have all triangular faces!

    triInfo = numpy.asarray([t[0] for t in tris], dtype=numpy.int32)
    nrmInfo = numpy.asarray([t[4] for t in tris]) # face normals as unit vectors
    cntInfo = numpy.asarray([t[2] for t in tris]) # center of face

    snrmInfo = numpy.asarray([t[1] for t in tris]) # scaled face normals, magnitude is area!

    # compute the vertex normals
    verts = numpy.zeros((1), dtype=numpy.int32)
    for t in triInfo: verts = numpy.append(verts, t, axis=0)
    verts = verts[1:]
    verts = numpy.unique(verts) # these are the surface verts
    vnormals = numpy.zeros((verts.shape[0],3))
    for i in range(verts.shape[0]): # loop over verts

        # find all tris that contain this
        for j in range(triInfo.shape[0]):
            t = triInfo[j]
            if not verts[i] in t: continue

            vnormals[i] += snrmInfo[j]
            #print(verts[i],t)

        vnormals[i] /= numpy.linalg.norm(vnormals[i])
        


    # now find edge means and normals
    edges = numpy.zeros((triInfo.shape[0]*3, 2), dtype=numpy.int32)
    
    c = 0
    for t in triInfo:
        edges[c] = [t[0], t[1]]; c += 1;
        edges[c] = [t[0], t[2]]; c += 1
        edges[c] = [t[1], t[2]]; c += 1
    
    edges = numpy.unique(edges, axis=0)
    vdict = dict(zip(verts, vnormals))

    edgeInfo = numpy.zeros((edges.shape[0], 6))
    for i in range(edges.shape[0]):
        e = edges[i]
        # find all triangles that have this edge
        enrm = numpy.zeros((3))
        for j in range(triInfo.shape[0]):
            t = triInfo[j]
            if (e[0] == t[0] and e[1] == t[1]) or (e[0] == t[0] and e[1] == t[2]) or (e[0] == t[1] and e[1] == t[2]):
                enrm += snrmInfo[j]
        enrm = enrm / numpy.linalg.norm(enrm)
                
        edgeInfo[i,0:3] = 0.5 * (xyz[e[0]] + xyz[e[1]])
        edgeInfo[i,3:] = enrm
        

    # TODO: chose how to combine the information in the output
    #
    # Nt = number of triangular faces on the surface
    # Ne = number of edges on the surface
    # Nv = number of atoms on the surface
    # 
    #
    # triInfo: Nt x 3 matrix, on each row -> three indexes of the atoms that make up the face, sorted by index!
    # nrmInfo: Nt x 3 matrix, on each row -> nx,ny,nz components of the normal vector of each face (normalised)
    # snrmInfo:Nt x 3 matrix, on each row -> nx,ny,nz components of the normal vector of each face (NOT normalised)
    # cntInfo: Nt x 3 matrix, on each row -> x,y,z components of the face center point
    #
    # verts: Nv list, each element is the index of a surface atom
    # vnormals: Nv x 3 matrix, on each row -> nx,ny,nz components of a surface atom normal vector (normalised)
    #
    # edges: Ne x 2 matrix, on each row there are the indexes of the two atoms and the ends
    # edgeInfo: Ne x 6 matrix, each row -> x,y,z,nx,ny,nz components of the edge midpoint position and normal (normalised)
    #
    #
    # these files are temporary for debug view in mathematica
    #numpy.savetxt("tris.dat", triInfo, "%d")
    #numpy.savetxt("normals.dat", nrmInfo)
    #numpy.savetxt("centers.dat", cntInfo)
    #numpy.savetxt("verts.dat", xyz[verts])
    #numpy.savetxt("vnormals.dat", vnormals)
    #numpy.savetxt("einfo.dat", edgeInfo)

    summary_dict = {
        "ids_surface_atoms" : verts,
        "positions_surface_atoms" : xyz[verts],
        "normals_surface_atoms" : vnormals,
        "ids_surface_edges" : edges,
        "centers_surface_edges" : edgeInfo[:, 0:3],
        "normals_surface_edges" : edgeInfo[:, 4:],
        "ids_surface_triangles" : triInfo,
        "centers_surface_triangles" : cntInfo,
        "normals_surface_triangles" : nrmInfo,
        }

    return summary_dict


if __name__ == "__main__":
    
    filename =   "../examples/example_structures/pt55.xyz"
    #filename = "../examples/example_structures/mos2.xyz"
    fxyz = open(filename,"r")
    
    n = int(fxyz.readline())
    fxyz.readline()

    atoms = numpy.zeros((n,3))
    for i in range(n):
        w = fxyz.readline().split()[1:]
        w = [float(x) for x in w]
        atoms[i] = w

    summary_dict = delaunator(atoms, 5.0)
    print(summary_dict)
    print(summary_dict["ids_surface_atoms"].shape, summary_dict["positions_surface_atoms"].shape,
        summary_dict["normals_surface_atoms"].shape)

    print("######################################")

    print(summary_dict["positions_surface_atoms"])
    print(summary_dict["ids_surface_atoms"])
    print(atoms)
    print(atoms.shape)


    import ase, ase.io 
    from ase.visualize import view

    ase_atoms = ase.io.read(filename)

    atomic_numbers = numpy.ones(len(ase_atoms))
    atomic_numbers[summary_dict["ids_surface_atoms"]] = 6
    ase_atoms.set_atomic_numbers(atomic_numbers)

    view(ase_atoms)