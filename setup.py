import platform
from setuptools import setup

# Determine the right package based on the user's system
onnx_dependency = 'onnxruntime-silicon' if platform.system(
) == 'Darwin' and platform.machine() == 'arm64' else 'onnxruntime-gpu==1.15.1'

setup(
    name='WALDO',
    version='2.5',
    description=
    'W.A.L.D.O. Whereabouts Ascertainment for Low-lying Detectable Objects',
    author='Stephan Sturges',
    author_email='stephan.sturges@gmail.com',
    url='https://github.com/stephansturges/WALDO',
    install_requires=[
        'numpy==1.23.5', 'opencv_contrib_python==4.5.5.62',
        'opencv_python_headless==4.7.0.72', 'Pillow==10.0.0',
        'Requests==2.31.0', 'torch==2.0.0', onnx_dependency
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        # Add more classifiers as needed
    ])
