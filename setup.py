from setuptools import setup, find_packages
import sys
import os


with open("requirements.txt", encoding="utf-8") as req_fp:
  install_requires = req_fp.readlines()

setup(
  name='pytorch_neat',
  version='0.0.1',
  description='local test setup',
  author='Some guys at uberresearch and this ai nick dude',
  url='https://github.com/ai-nick/PyTorch-NEAT',
  license='Apache License',
  #install_requires=install_requires, #uncomment to actually install this 
  packages=find_packages(exclude=[]),
  python_requires='>=3.5',
  project_urls={
    'Source': 'https://github.com/ai-nick/PyTorch-NEAT',
  },
  entry_points={
    'console_scripts': [
    ],
  }
)
