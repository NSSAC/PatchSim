"""Setup."""

from setuptools import setup

package_name = "PatchSim"
description = "A system for simulating the metapopulation SEIR model"


with open("README.md", "r") as fh:
    long_description = fh.read()

classifiers = """
Development Status :: 3 - Alpha
Environment :: Console
Intended Audience :: Developers
Intended Audience :: Science/Research
Intended Audience :: Healthcare Industry
License :: OSI Approved :: MIT License
Operating System :: POSIX :: Linux
Programming Language :: Python :: 3.7
Topic :: Scientific/Engineering
""".strip().split("\n")

setup(
    name=package_name,
    description=description,

    author="Srini Venkatramanan",
    author_email="srini@virginia.edu",

    long_description=long_description,
    long_description_content_type="text/markdown",

    # packages=[package_name],
    py_modules = ["pathchsim"],

    use_scm_version=True,
    setup_requires=['setuptools_scm'],

    install_requires=[
        "numpy",
        "pandas",
        "click",
    ],

    url="http://github.com/nssac/pathchsim",
    classifiers=classifiers
)
