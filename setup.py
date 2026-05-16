from setuptools import setup, find_packages

setup(
    name="cross_device_identity_resolution",
    version="1.0.0",
    description="Privacy-preserving cross-device identity resolution and contextual audience targeting",
    author="Aswin Gunasekaran",
    author_email="aswinguna75@gmail.com",
    url="https://github.com/Aswinguna/cross-device-identity-resolution",
    packages=find_packages(where="."),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "sentence-transformers>=2.2.2",
        "spacy>=3.6.0",
        "mlflow>=2.5.0",
        "dash>=2.11.0",
        "dash-bootstrap-components>=1.5.0",
        "plotly>=5.15.0",
        "sqlalchemy>=2.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "xgboost": ["xgboost>=1.7.0"],
        "mysql": ["pymysql>=1.1.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
    entry_points={
        "console_scripts": [
            "run-pipeline=pipeline:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: Log Analysis",
    ],
)
