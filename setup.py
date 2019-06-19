from setuptools import setup, find_packages

setup(
    name='Kostas_Simulator',
    version='0.1',
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'matplotlib', 'opencv-python'],
    scripts=['kostas.py']
)