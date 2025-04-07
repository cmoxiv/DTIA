
from setuptools import setup, find_packages

def read_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()
    
setup(
    name='dtia',
    version='0.1.1',
    description='Decision Tree Insight Analysis Tool',
    url='https://github.com/karim/dtia',
    author='Karim Hossny',
    author_email='k.hossny@kth.se',
    license='BSD 2-clause',
    packages=find_packages(),
    install_requires=read_requirements(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux :: Windows :: MacOS',
        'Programming Language :: Python :: 3.8',
    ],
)
