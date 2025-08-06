# This setup.py file is used to package and distribute the electrodiffusion library.
# It defines the package metadata, dependencies, and installation instructions using setuptools.
# The script allows users to install the library via 'pip install .' or 'pip install -e .' for development mode.
# It includes classifiers for PyPI, specifies Python version requirements, and handles package data if needed.
# Long description is read from README.md for detailed package information.
# This implementation is standard for Python packages and draws from best practices; it does not directly reference
# Claude's code artifacts but ensures the library is installable as a cohesive unit, compatible with the modular
# structure (e.g., subpackages like models and simulations).

import os
from setuptools import setup, find_packages

# Read the long description from README.md
def read_readme():
    """
    Reads the content of README.md for the long_description in setup.
    
    Returns:
        str: Content of README.md or empty string if not found.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    readme_path = os.path.join(here, 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Package metadata
NAME = 'electrodiffusion'
VERSION = '0.1.0'  # Initial version; update with semantic versioning (major.minor.patch)
DESCRIPTION = 'A Python library for modeling electrodiffusion in ion channels using score-based diffusion and SDE frameworks.'
LONG_DESCRIPTION = read_readme()
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'  # For README.md formatting on PyPI
AUTHOR = 'Your Name or Organization'  # Replace with actual author (e.g., based on Claude's contributions)
AUTHOR_EMAIL = 'your.email@example.com'  # Replace with contact email
URL = 'https://github.com/your-repo/electrodiffusion'  # Replace with actual repository URL
LICENSE = 'MIT'  # Or appropriate license; MIT is common for open-source

# Required dependencies (core libraries needed to run the code)
# Versions are pinned loosely to allow flexibility; adjust based on testing
INSTALL_REQUIRES = [
    'torch >= 2.0.0',  # Core tensor library for models and simulations
    'numpy >= 1.20.0',  # Numerical computations
    'matplotlib >= 3.5.0',  # Visualization in utils.viz
    'seaborn >= 0.11.0',  # Enhanced plotting
    'scipy >= 1.7.0',  # Signal processing (e.g., welch in examples)
    'argparse >= 1.1',  # Standard, but listed for completeness
    # Optional for full functionality (not required, but recommended)
    # 'mdanalysis >= 2.0.0',  # For real MD data loading in utils.data
    # 'gudhi >= 3.4.0',  # For persistent homology in utils.topology
    # 'pytest >= 7.0.0',  # For running tests
]

# Extra dependencies (e.g., for development or optional features)
EXTRAS_REQUIRE = {
    'dev': ['pytest >= 7.0.0', 'black >= 22.0.0', 'flake8 >= 4.0.0'],  # Development tools
    'md': ['mdanalysis >= 2.0.0'],  # For MD trajectory loading
    'topology': ['gudhi >= 3.4.0'],  # For topology utils
    'all': ['mdanalysis >= 2.0.0', 'gudhi >= 3.4.0', 'pytest >= 7.0.0']  # All extras
}

# Classifiers for PyPI (helps with discoverability)
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',  # Adjust based on maturity
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Topic :: Scientific/Engineering :: Physics',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Operating System :: OS Independent',
]

# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,
    packages=find_packages(exclude=['tests', 'examples']),  # Automatically find packages, exclude tests/examples
    include_package_data=True,  # Include non-code files (e.g., data files if any)
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=CLASSIFIERS,
    python_requires='>=3.8',  # Minimum Python version based on features (e.g., dataclasses)
    # Entry points for scripts (e.g., console scripts if needed)
    entry_points={
        'console_scripts': [
            # 'electrodiffusion-tool = electrodiffusion.some_module:main',  # Example; add if tools are defined
        ]
    },
)
