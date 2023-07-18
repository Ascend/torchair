"""Basic configration for setting up NPU device"""

from setuptools import setup
from setuptools import find_packages

setup(name='torchair',
      version='0.1',
      description='em...',
      long_description='em!!!',
      packages=find_packages(),
      include_package_data=True,
      ext_modules=[],
      install_requires=[
        'protobuf >= 3.13, < 4',
      ],
      zip_safe=False)
