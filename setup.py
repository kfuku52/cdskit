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
    install_requires=['numpy', 'biopython'],
    extras_require={'test': ['pytest', 'pytest-cov']},
    python_requires='>=3.9',
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    scripts=['cdskit/cdskit'],
)
