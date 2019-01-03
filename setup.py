from setuptools import setup, find_packages, Distribution, Extension

extensions = [Extension("lib.libclusgeo3",
                       ["src/clusGeo3.c"],
                       include_dirs=["src"],
                       libraries=["m"],
                       extra_compile_args=["-O3", "-std=c99"]
              )]

if __name__=="__main__":
  setup(name="clusgeo",
      url="https://github.com/SINGROUP/SOAPLite",
      version="4.0.3",
      description=("CLUSter GEOmetry tool for surface chemistry."), author="Eiaki V. Morooka", author_email="eiaki.morooka@aalto.fi",
      packages = find_packages(),
      install_requires =["numpy>=1.14.2",
      "scipy",
      "dscribe",
      "ase",]
      , python_requires='>=2.2, <4', keywords="descriptor materials science machine learning soap local environment materials physics symmetry reduction adsorption sites",
      license="LGPLv3", classifiers=['Topic :: Scientific/Engineering :: Physics', 'Operating System :: POSIX :: Linux' ,'Topic :: Scientific/Engineering :: Chemistry','Topic :: Scientific/Engineering :: Artificial Intelligence','License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)', 'Development Status :: 4 - Beta', 'Intended Audience :: Science/Research','Intended Audience :: Religion', 'Intended Audience :: Education','Intended Audience :: Developers','Programming Language :: Python','Programming Language :: C' ],
      package_data={
        'clusgeo':['lib/libclusgeo3.so']},
      ext_modules=extensions)


