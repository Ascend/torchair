from setuptools import setup, find_packages

setup(name='inductor-npu-ext',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch>=2.8.0',
        'importlib-metadata',
    ])
