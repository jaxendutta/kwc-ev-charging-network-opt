from setuptools import setup, find_packages

# Read the requirements from [requirements.txt]
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='kwc_ev_opt',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            # Define any command-line scripts here
        ],
    },
)