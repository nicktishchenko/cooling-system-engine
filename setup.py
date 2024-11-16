from setuptools import setup, find_packages

setup(
    name="coolsys",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "nltk>=3.8.0",
        "pandas>=2.0.0",
        "pyodbc>=4.0.39",
        "spellchecker>=0.7.1",
        "jupyter>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ]
    },
    python_requires=">=3.11",
    author="Nikolay Tishchenko",
    author_email="nicktishchenko@gmail.com",
    description="Cooling System Analysis Engine - NLP-based maintenance report analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nicktishchenko/cooling-system-engine",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
)
