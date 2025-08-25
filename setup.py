from setuptools import setup, find_packages
from typing import List


HYPEN_E_Dot = '-e .'

def get_requirements(file_path:str) -> List[str]:
    """ 
    This function reads the requirements from a file and returns them as a list.
    """
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements= [req.replace('\n', '') for req in requirements]

    # Remove any hyphen-e dot requirement if it exists
        if HYPEN_E_Dot in requirements:
            requirements.remove(HYPEN_E_Dot)
    return [req.strip() for req in requirements if req.strip()]
    # return requirements

setup(
    name='customer-churn-prediction',
    version='0.0.1',
    author='Sonalika Singh',
    author_email='singhsonalika5@gmail.com',
    description='Customer Churn Prediction End-to-End ML Project',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
