from setuptools import setup, find_packages

setup(
    name="sccvi_impl",
    version="0.1",
    packages=find_packages(where="sccvi_impl/src"),
    package_dir={"": "sccvi_impl/src"},
    python_requires=">=3.8",
    install_requires=[
        # Dependencies are already in requirements.txt
    ],
)
