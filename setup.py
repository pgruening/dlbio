from setuptools import setup, find_packages

setup(
    name='DLBio',
    version='0.0.1',
    packages=['DLBio'],
    url='https://github.com/pgruening/dlbio',
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'scikit-learn',
        'tkinter',
        'matplotlib',
        'pandas',
        'opencv-python',
        'recordtype'
      ]
)
