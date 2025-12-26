"""
Setup script for GP-MPC Rocket Landing package.

This file provides backwards compatibility with older pip versions.
The main configuration is in pyproject.toml.
"""

from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="gp-mpc-rocket-landing",
        version="0.1.0",
        packages=find_packages(),
        python_requires=">=3.9",
        install_requires=[
            "numpy>=1.21.0",
            "scipy>=1.7.0",
            "matplotlib>=3.4.0",
            "pyyaml>=6.0",
        ],
        extras_require={
            "dev": [
                "pytest>=7.0",
                "pytest-cov>=4.0",
                "black>=23.0",
                "isort>=5.0",
                "mypy>=1.0",
            ],
            "optimization": [
                "casadi>=3.6.0",
                "osqp>=0.6.0",
            ],
        },
        author="GP-MPC Project Team",
        description="Gaussian Process enhanced MPC for rocket landing",
        long_description=open("README.md").read(),  # noqa: SIM115
        long_description_content_type="text/markdown",
        license="MIT",
        url="https://github.com/your-org/gp-mpc-rocket-landing",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering",
        ],
    )
