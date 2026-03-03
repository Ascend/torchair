from setuptools import setup, find_packages

setup(name='torchair',
      version='0.1',
      description='TorchAir',
      long_description='Torch Ascend Intermediate Representation',
      packages=find_packages(),
      include_package_data=True,
      ext_modules=[],
      install_requires=[
        'protobuf >= 3.13, < 4',
      ],
      zip_safe=False)
