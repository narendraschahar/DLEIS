# setup.py

from setuptools import setup, find_packages
import os
from typing import List

def read_requirements(filename: str) -> List[str]:
    """Read requirements from requirements.txt"""
    with open(os.path.join(os.path.dirname(__file__), filename)) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def get_version() -> str:
    """Get package version from src/__init__.py"""
    with open(os.path.join('src', '__init__.py')) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip("'").strip('"')
    raise RuntimeError('Version not found')

setup(
    name="deep-steganography",
    version=get_version(),
    description="Deep Learning-Empowered Image Steganography",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author="Narendra Kumar Chahar",
    author_email="write2nschahar@gmail.com",
    url="https://github.com/narendraschahar/DLEIS",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=read_requirements('requirements.txt'),
    extras_require={
        'dev': [
            'pytest>=6.2.5',
            'pytest-cov>=2.12.1',
            'black>=21.7b0',
            'isort>=5.9.3',
            'flake8>=3.9.2',
            'mypy>=0.910',
        ],
        'docs': [
            'sphinx>=4.1.2',
            'sphinx-rtd-theme>=0.5.2',
            'sphinx-autodoc-typehints>=1.12.0',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Security :: Cryptography',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'train-stego=scripts.train:main',
            'evaluate-stego=scripts.evaluate:main',
            'generate-results=scripts.generate_results:main',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/narendraschahar/DLEIS/issues',
        'Source': 'https://github.com/narendraschahar/DLEIS',
        'Documentation': 'https://dleis.readthedocs.io/',
    },
    keywords=[
        'deep-learning',
        'steganography',
        'image-processing',
        'security',
        'pytorch',
        'vision-transformer',
        'wavelet-transform',
        'information-hiding',
    ],
)