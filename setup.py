from setuptools import setup, find_packages

setup(
    name="torchao-to-compressed-tensors",
    version="0.2.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    entry_points={
        "console_scripts": [
            "torchao-to-ct = torchao_to_compressed_tensors.adapter:main",
        ],
    },
)
