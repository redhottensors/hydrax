from setuptools import setup, Extension

setup(
    name="hydrax",
    version="0.9",
    ext_modules=[
        Extension("hydrax._trackedbuffer", ['hydrax/trackedbuffer.c'])
    ],
    packages=['hydrax'],
)
