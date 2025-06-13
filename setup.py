from setuptools import setup, find_packages

setup(
    name="End-to-End-Insurance-Risk-Analytics-Predictive-Modeling",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
    ],
    author="Abel Getahun",
    author_email="abelgetahun66@gmail.com.com",
    description="Insurance Risk Analytics and Predictive Modeling",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/abeelgetahun/End-to-End-Insurance-Risk-Analytics-Predictive-Modeling.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)