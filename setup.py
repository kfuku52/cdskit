from setuptools import setup, find_packages

setup(
        name             = 'cdskit',
        version          = "0.2",
        description      = 'Tools for processing protein-coding sequences',
        license          = "BSD 3-clause License",
        author           = "Kenji Fukushima",
        author_email     = 'kfuku52@gmail.com',
        url              = 'https://github.com/kfuku52/cdskit.git',
        keywords         = '',
        packages         = find_packages(),
        install_requires = ['numpy','biopython',],
        scripts          = ['cdskit/cdskit',],
)