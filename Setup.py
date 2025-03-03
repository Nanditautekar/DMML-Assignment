from setuptools import find_packages, setup
from typing import List 

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    ## This function will return the list of reqirements ##

    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","")for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
name='DMML Assignment',
version='0.0.1',
author_email='Shinchan02022000@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)
from setuptools import setup, find_packages

from setuptools import setup, find_packages

setup(
    name="src",
    version="0.1",
    packages=find_packages(),
)