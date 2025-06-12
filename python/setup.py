from setuptools import setup

setup(name='npu_extension_for_inductor',
      version='0.0.1',
      entry_points={
            'console_scripts': [
                  'asc_pgo = npu_extension_for_inductor.pgo.asc_pgo:run',
            ],
      }
)
