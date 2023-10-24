import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


setup(
    ext_modules=[
        Pybind11Extension(
            name="_igwas",
            sources=sorted(glob.glob("indirect_gwas/src/*.cpp")),
            include_dirs=[
                "indirect_gwas/src",
                "extern/math-boost-1.83.0/include",
                "extern/eigen-3.4.0",
            ],
            language="c++",
        )
    ],
    zip_safe=False,
    cmdclass={"build_ext": build_ext},
)
