import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="symreg",
    version="0.0.4",
    author="Dan Gheorghe Haiduc",
    author_email="danuthaiduc@gmail.com.com",
    description="A Symbolic Regression engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danuker/symreg",
    packages=setuptools.find_packages(exclude=('tests', 'tests.*')),
    install_requires=['numpy', 'orderedset'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
