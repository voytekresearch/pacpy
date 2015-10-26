"""pacpy setup script"""
from setuptools import setup, find_packages

setup(
    name='pacpy',
    version='1.0',
    description="A module to calculate phase-amplitude coupling",
    long_description = """
    A module to calculate phase-amplitude coupling of neural time series.
    """,
    url='https://github.com/voytekresearch/pacpy',
    download_url = 'https://github.com/voytekresearch/pacpy/archive/v1.0.tar.gz',
    author='The Voytek Lab',
    author_email='voyteklab@gmail.com',
    license='MIT',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7'
    ],
    keywords='neuroscience spectral phase-amplitude',
    packages=find_packages(),
    install_requires=['numpy>=1.8.0','scipy>=0.15.1'],
    extras_require={'test': ['pytest']},
    package_data={
        '': ['tests/*.npy', 'tests/test_*py'],
    },
    test_suite="tests"
)
