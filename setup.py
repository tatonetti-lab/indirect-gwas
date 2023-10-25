import glob
import os

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


eigen_include_path = os.environ.get('EIGEN_INCLUDE_PATH', '/usr/include')
boost_math_include_path = os.environ.get('BOOST_MATH_INCLUDE_PATH', '/usr/include')
csv_include_path = os.environ.get('CSV_INCLUDE_PATH', '/usr/include')

setup(
    ext_modules=[
        Pybind11Extension(
            name="indirect_gwas._igwas",
            sources=sorted(glob.glob("indirect_gwas/src/*.cpp")),
            include_dirs=[
                "indirect_gwas/src",
                boost_math_include_path,
                eigen_include_path,
                csv_include_path,
            ],
            language="c++",
        )
    ],
    zip_safe=False,
    cmdclass={"build_ext": build_ext},
)
