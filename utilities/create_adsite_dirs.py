import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import numpy as np 
import ase, ase.io
import os, argparse, time
from ctypes import *
import soaplite
import pathlib
import cluskit
from os.path import basename

if __name__ == "__main__":
    t0_total = time.time()

    ### INPUT ###
    parser = argparse.ArgumentParser(description=
        'Give input xyz filename ')
    parser.add_argument('arguments', metavar='args', type=str, nargs='+',
                                   help='[filename]')
    args = parser.parse_args()
    print("Passed arguments:", args.arguments)
    if len(args.arguments) < 1:
        print('Not enough arguments')
        exit(1)
    infilename = args.arguments[0]
    rootname = basename(infilename)
    rootname = rootname.replace(".xyz", "")
    ### PROCESS ###

    atoms = ase.io.read(infilename)
    cluskit.write_all_sites(atoms, rootname)

    t1_total = time.time()
    print("Total run time:", str(t1_total - t0_total))


