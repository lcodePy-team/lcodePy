import re
from pathlib import Path

from distutils.command.install import INSTALL_SCHEMES
from setuptools import find_packages, setup


def get_parameter(parameter_name: str, init_file_path: Path = None) -> str:
    """
    Retrieve the value of a specified parameter from the lcode/__init__.py file.
    
    Args:
        parameter_name (str): The name of the parameter to retrieve.
        init_file_path (Path, optional): The path to the __init__.py file. Defaults to the lcode/__init__.py file in the current directory.
    
    Returns:
        str: The value of the specified parameter as a string.
    
    Raises:
        RuntimeError: If the parameter string cannot be found or the file cannot be accessed.
    """
    if init_file_path is None:
        init_file_path = Path(__file__).resolve().parent / 'lcode' / '__init__.py'
    
    try:
        text = Path(init_file_path).read_text()
        match = re.search(fr'{parameter_name} = [\"\'](.*?)[\"\']', text)
        if not match:
            raise RuntimeError(f'Cannot find {parameter_name} in {init_file_path}')
        return match.group(1)
    except OSError:
        raise RuntimeError(f'Cannot access {init_file_path}')

with open('README.md', 'r') as f:
    long_description = f.read()

for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']

with open('requirements.txt', 'r') as f:
    requirements = f.read().split('\n')

if __name__ == '__main__':
    setup(
        name='lcode',
        version=get_parameter('__version__'),
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
        package_data={'lcode': ['examples/*.py']},
        include_package_data=True,
        classifiers=[
          'Programming Language :: Python',
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Physics',
          'Environment :: Console',
          'Operating System :: OS Independent'
        ],
        license ='BSD-3-Clause-extended', 
        license_files = ('LICENSE',),
        keywords=['plasma wakefield acceleration', 
            'quasistatic approximation',
            'numerical simulation'],
)
    
