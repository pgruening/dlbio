from setuptools import setup, find_packages

setup(
    name='DLBio',
    version='0.0.1',
    packages=['DLBio'],
    url='https://github.com/pgruening/dlbio',
    install_requires=[
        'scikit-learn',
        'matplotlib',
        'pandas',
        'opencv-python',
        'recordtype'
      ]
)
