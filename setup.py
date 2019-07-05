from setuptools import setup, find_packages, Distribution, Extension

extensions = [Extension("lib.libcluskit3",
                       ["src/cluskit.c"],
                       include_dirs=["src"],
                       libraries=["m"],
                       extra_compile_args=["-O3", "-std=c99"]
              )]

if __name__=="__main__":
  setup(name="cluskit",
      url="https://github.com/SINGROUP/SOAPLite",
      version="4.1.9",
      description=("(Nano)CLUSter KIT for surface chemistry."), author="Marc Jaeger", author_email="marc.jager@aalto.fi",
      packages = find_packages(),
      install_requires =["numpy>=1.14.2",
      "scipy",
      "dscribe>=0.2.7",
      "ase",]
      , python_requires='>=2.2, <4', keywords="descriptor materials science machine learning soap local environment materials physics symmetry reduction adsorption sites",
      license="LGPLv3", classifiers=['Topic :: Scientific/Engineering :: Physics', 'Operating System :: POSIX :: Linux' ,'Topic :: Scientific/Engineering :: Chemistry','Topic :: Scientific/Engineering :: Artificial Intelligence','License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)', 'Development Status :: 4 - Beta', 'Intended Audience :: Science/Research','Intended Audience :: Religion', 'Intended Audience :: Education','Intended Audience :: Developers','Programming Language :: Python','Programming Language :: C' ],
      package_data={
        'cluskit':['lib/libcluskit.so']},
      ext_modules=extensions)


