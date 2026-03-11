from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'get_sensor_data',
        ['get_sensor_data.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++17']
    ),
]

setup(
    name='get_sensor_data',
    version='0.0.1',
    author='CaptainYin',
    ext_modules=ext_modules,
    setup_requires=['pybind11'],
)

