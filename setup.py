import os
import setuptools
import CONSTANTS
from setuptools import find_packages
import glob

readme = "README.md"
requirement_path = CONSTANTS.Requirements_path
readme_path = os.path.join(os.path.dirname(__file__), readme)
data_files_01 = glob.glob('petdata/2015/08/01' + '/*.jpg', recursive=True)
data_files_02 = glob.glob('petdata/2015/08/02' + '/*.jpg', recursive=True)
data_files_03 = glob.glob('petdata/2015/08/03' + '/*.jpg', recursive=True)
data_files_04 = glob.glob('petdata/2015/08/04' + '/*.jpg', recursive=True)
data_files_05 = glob.glob('petdata/2015/08/05' + '/*.jpg', recursive=True)
data_files_06 = glob.glob('petdata/2015/08/06' + '/*.jpg', recursive=True)
data_files_07 = glob.glob('petdata/2015/08/07' + '/*.jpg', recursive=True)
data_files_08 = glob.glob('petdata/2015/08/08' + '/*.jpg', recursive=True)
data_files_09 = glob.glob('petdata/2015/08/09' + '/*.jpg', recursive=True)
with open(readme, "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(requirement_path) as f:
    install_requires = f.read().splitlines()

pkgs = find_packages(exclude=['tests.*', 'tests'])

setuptools.setup(
    name="augmentation_engine",
    version="0.1.0",
    author="Shreejaa Talla",
    author_email="shreejaa.talla@gmail.com",
    description="Solar Filaments data augmentation demo package",
    url="https://bitbucket.org/gsudmlab/augmentation_engine/src/master/",
    project_urls={
        "Source": "https://bitbucket.org/gsudmlab/augmentation_engine/src/master/",
    },
    packages = pkgs,
    package_dir={"filament_augmentation": "filament_augmentation"},
    package_data={
        '.': ['requirements.txt'],
        '':['petdata/2015']},
    install_requires=install_requires,
    py_modules=['CONSTANTS'],
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
)

