from setuptools import setup, find_packages
import os
import re

with open(os.path.join('csubst', '__init__.py')) as f:
        match = re.search(r'__version__\s+=\s+(.*)', f.read())
version = str(ast.literal_eval(match.group(1)))

setup(
        name             = 'cdskit',
        version          = version,
        description      = 'Tools for processing codon sequences',
        license          = "BSD 3-clause License",
        author           = "Kenji Fukushima",
        author_email     = 'kfuku52@gmail.com',
        url              = 'https://github.com/kfuku52/cdskit.git',
        keywords         = 'codon sequences',
        packages         = find_packages(),
        install_requires = ['numpy','biopython',],
        scripts          = ['cdskit/cdskit',],
)