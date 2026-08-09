import ast
import re
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
INIT_FILE = ROOT / 'cdskit' / '__init__.py'
README_FILE = ROOT / 'README.md'

match = re.search(r'__version__\s+=\s+(.*)', INIT_FILE.read_text(encoding='utf-8'))
if match is None:
    raise RuntimeError(f'Could not find __version__ in {INIT_FILE}')

version = str(ast.literal_eval(match.group(1)))
long_description = README_FILE.read_text(encoding='utf-8')

TEST_REQUIRES = [
    'pytest>=7',
]
COVERAGE_REQUIRES = [
    'pytest-cov>=5',
    'pytest-xdist>=3',
]
QUALITY_REQUIRES = [
    'mypy>=1.10',
    'ruff>=0.6',
]
BUILD_REQUIRES = [
    'build>=1',
]
ML_REQUIRES = [
    'torch>=2.2',
    'scikit-learn>=1.4',
    'transformers>=4.40',
]

setup(
    name='cdskit',
    version=version,
    description='Tools for processing codon sequences',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='BSD-3-Clause',
    author='Kenji Fukushima',
    author_email='kfuku52@gmail.com',
    url='https://github.com/kfuku52/cdskit',
    project_urls={
        'Repository': 'https://github.com/kfuku52/cdskit',
        'Issues': 'https://github.com/kfuku52/cdskit/issues',
    },
    keywords='codon sequences',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.23',
        'biopython>=1.80',
        'matplotlib>=3.6',
    ],
    extras_require={
        'test': TEST_REQUIRES,
        'coverage': COVERAGE_REQUIRES,
        'quality': QUALITY_REQUIRES,
        'build': BUILD_REQUIRES,
        'ml': ML_REQUIRES,
        'dev': (
            TEST_REQUIRES
            + COVERAGE_REQUIRES
            + QUALITY_REQUIRES
            + BUILD_REQUIRES
            + ML_REQUIRES
        ),
    },
    python_requires='>=3.10',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    entry_points={
        'console_scripts': [
            'cdskit=cdskit.cli:main',
        ],
    },
)
