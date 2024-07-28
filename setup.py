from setuptools import setup, Extension

module = Extension("example", sources=["main.cpp"])

setup(
    name="example",
    version="1.0",
    ext_modules=[module],
)
