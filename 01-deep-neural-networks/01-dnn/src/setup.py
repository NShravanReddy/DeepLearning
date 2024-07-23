import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="micrograd",
    version="0.1.0",
    author="N Shravan Reddy",
    author_email="nshravanreddy6@gmail.com",
    description="Micrograd",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NShravanReddy/DeepLearning/tree/main/01-deep-neural-networks/01-dnn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)