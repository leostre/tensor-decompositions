from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# print(find_packages(exclude=["tests*", "experiments*", "docs*", "build*", "dist*"]))

setup(
    name="tdecomp",
    version="0.2.18",
    description="Library for tensor and matrix decompositions applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Leon Strelkov",
    author_email="strelllleo@mail.ru",
    url="https://github.com/leostre/tensor-decompositions.git",
    packages=find_packages(exclude=["tests*", "experiments*", "docs*", "build*", "dist*"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "tensorly",
        "torch",
    ],
    license=' BSD License',
    keywords=[
        "tensor decompositions",
        "matrix decompositions", 
        "randomized algorithms"
    ],
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)