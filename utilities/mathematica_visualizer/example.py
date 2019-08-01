if __name__ == '__main__':
    import ase
    from cluskit import Support
    from ase.build import fcc111
    slab = fcc111('Al', size=(3,3,3), vacuum=10.0)
    slab = ase.build.bcc100('Fe', size=(3,3,3), vacuum=10.0)
    #diamond = ase.build.diamond111('C', size = (3,3,6), a=None, vacuum=8.0, orthogonal=False, periodic=True)
    #slab = diamond
    print(slab.get_cell())
    print(len(slab))
    sup = Support(slab)
    #sup = Support(slab, max_bondlength = 2.7)
    #sup.max_bondlength = 2.3
    sa = sup.get_surface_atoms()
    print("surface atoms", sa)
    print("surface atoms", len(sa))

    print(sup.adsorption_vectors[1].shape,
        sup.adsorption_vectors[2].shape,
        sup.adsorption_vectors[3].shape)
    reduced_surface_atoms = sup.reduce_surface_atoms(z_direction = 1, z_cut =11.0)
    print(reduced_surface_atoms, len(reduced_surface_atoms))
    print("reduced_surface_atoms")
    print(sup.adsorption_vectors[1].shape,
        sup.adsorption_vectors[2].shape,
        sup.adsorption_vectors[3].shape)

    from ase.visualize import view
    view(slab[reduced_surface_atoms])
    view(slab[sa])
    view(slab)
    ase.io.write("structure.xyz", slab)
