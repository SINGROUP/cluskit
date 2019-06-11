
# currently not used, but might come in handy at some point
def _format_ase2cluskit(obj, all_atomtypes=[]):
    """ Takes an ase Atoms object and returns numpy arrays and integers
    which are read by the internal cluskit. Apos is currently a flattened
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


