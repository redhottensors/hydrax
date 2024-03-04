from setuptools import setup, Extension

setup(
    name="hydrax",
    version="0.9",
    packages=["hydrax"],
    ext_modules=[
        Extension("hydrax._trackedbuffer", ["hydrax/trackedbuffer.c"])
    ],
)
