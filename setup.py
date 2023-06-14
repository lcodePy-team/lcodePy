from setuptools import setup, find_packages
from lcode import __version__
    
with open('README.md', 'r') as f:
    long_description = f.read()


with open('requirements.txt', 'r') as f:
    requirements = f.read().split('\n')

if __name__ == '__main__':
    setup(
        name='lcode',
        version=__version__,
        author='lcodePy-team',
        author_email='team@lcode.info',
        description='LCODE is a free software for simulation' + 
                'plasma wakefield acceleration based on QSA.',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://lcode.info',
        download_url='https://github.com/lcodePy-team/lcodePy',
        packages=find_packages(),
        install_requires=requirements,
        include_package_data=True,
        classifiers=[
          'Programming Language :: Python',
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          "Topic :: Scientific/Engineering :: Physics",
          'Environment :: Console',
          'Operating System :: OS Independent'
        ],
        license ='BSD-3-Clause-extended', 
        license_files = ('LICENSE',),
        keywords=['plasma wakefield acceleration', 
            'quasistatic approximation',
            'numerical simulation'],
)
    
