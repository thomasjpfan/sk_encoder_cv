from setuptools import setup
from setuptools import find_packages

setup(
    name="sk_encoder_cv",
    version="0.0.1",
    author="Thomas J. Fan",
    packages=find_packages("sk_encoder_cv"),
    install_requires=["pandas==1.2.0", "scikit-learn==0.24.0", "pytest==6.2.1"],
    license="MIT",
)
