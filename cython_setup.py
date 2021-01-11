from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension("env.observations.TreeObsForRailEnv", ["env/observations/TreeObsForRailEnv.pyx"], ),
    Extension("env.observations.SimpleObservation", ["env/observations/SimpleObservation.pyx"], ),
]

setup(
    name='Hi',
    ext_modules=cythonize(ext_modules),
)
