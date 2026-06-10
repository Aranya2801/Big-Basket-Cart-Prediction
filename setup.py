"""
BigBasket Cart Prediction — Package Setup
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="bigbasket-cart-prediction",
    version="2.0.0",
    author="Aranya2801",
    author_email="aranya@example.com",
    description="AI-Powered Grocery Cart Prediction System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Aranya2801/Big-Basket-Cart-Prediction",
    packages=find_packages(exclude=["tests*", "notebooks*", "docs*"]),
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "bb-predict=bb_cli:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=["recommendation-system", "market-basket-analysis", "grocery",
              "association-rules", "collaborative-filtering", "streamlit"],
)
