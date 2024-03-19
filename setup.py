from setuptools import setup, Extension

setup(
    packages=["hydrax"],
    ext_modules=[
        Extension("hydrax._trackedbuffer", ["csrc/trackedbuffer.c"])
    ],
)
