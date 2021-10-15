from distutils.core import setup
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import Extension
import os
import numpy as np
from Cython.Build import cythonize
import cython_gsl

__version__ = '1.0'

gsl_funcs_ext = Extension('SWGSimulator.Tools.gsl_funcs',
                          ['SWGSimulator/Tools/gsl_funcs.pyx'],
                          libraries=cython_gsl.get_libraries(),
                          library_dirs=[cython_gsl.get_library_dir()],
                          include_dirs=[cython_gsl.get_include()],
                          extra_compile_args=['-fopenmp'],
                          extra_link_args=['-fopenmp']
                      )

pysla = Extension(name = 'SWGSimulator.Tools.pysla', 
                        sources = ['SWGSimulator/Tools/pysla.f90','SWGSimulator/Tools/sla.f'])


config = {'name':'SWGSimulator',
          'version':__version__,
          'packages':['SWGSimulator',
                      'SWGSimulator.SkyModel',
                      'SWGSimulator.SimObs',
                      'SWGSimulator.Tools',
                      'SWGSimulator.Telescope',
                      'SWGSimulator.Beams'],
          'ext_modules':cythonize([pysla,gsl_funcs_ext])}

#cythonize([pysla, gsl_funcs_ext], 
#                                  compiler_directives={'language_level':"3"})}



setup(**config)
