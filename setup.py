from pathlib import Path 
from setuptools import setup, find_packages

ROOT_DIR = Path(__file__).parent.resolve()

NAME = 'tdecomp'
VERSION = '0.1'

AUTHOR = 'Leon Strelkov'
AUTHOR_EMAIL = 'strelllleo@mail.ru'
LICENSE = "BSD-3-Clause"
SHORT_DESCRIPTION = 'Library for tensor and matrix decompositions applications'
README = (ROOT_DIR / 'README.md').read_text(encoding='utf-8')
URL = 'https://github.com/leostre/tensor-decompositions.git'
REQUIRES_PYTHON = '>=3.8'
EXCLUDED_PACKAGES = ['examples*', 'experiments*', 'tests*']
KEYWORDS = 'tensor decompositions, matrix decompositions, randomized algorithms'

def _read_lines(*names, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    lines = (ROOT_DIR
             .joinpath(*names)
             .read_text(encoding=encoding)
             .splitlines())
    return list(map(str.strip, lines))

def _extract_requirements(file_name):
    return [line for line in _read_lines(file_name) if line and not line.startswith('#')]


setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    python_requires=REQUIRES_PYTHON,
    description=SHORT_DESCRIPTION,
    long_description=README,
    install_requires=_extract_requirements('requirements.txt'),
    packages=find_packages(),
    license=LICENSE,
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords=KEYWORDS,
    include_package_data=True
)
