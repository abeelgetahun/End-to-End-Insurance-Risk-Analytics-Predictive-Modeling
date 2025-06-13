import os
from pathlib import Path

# --- Configuration ---
# Define the project structure as a dictionary.
# Keys are directories, and values are lists of subdirectories or files.
# Use an empty list for empty directories.
PROJECT_STRUCTURE = {
    "data": ["raw", "processed", "external"],
    "notebooks": [
        "01_data_exploration.ipynb",
        "02_eda_analysis.ipynb",
        "03_statistical_analysis.ipynb",
    ],
    "src": {
        "__init__.py": "",
        "data": ["__init__.py", "data_loader.py"],
        "features": ["__init__.py", "feature_engineering.py"],
        "visualization": ["__init__.py", "plots.py"],
        "utils": ["__init__.py", "helpers.py"],
    },
    "tests": ["__init__.py"],
    "reports": {"figures": [], "task1_eda_report.md": ""},
    "models": [],
    ".github": {"workflows": ["ci.yml"]},
    "docs": [],
    "README.md": "# Project Title\n\nProject Description.",
    ".gitignore": """
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.idea/
.vscode/
*.suo
*.ntvs*
*.njsproj
*.sln
*.sw?

# Data files
data/raw/
data/processed/

# Notebooks
.ipynb_checkpoints
""",
    "requirements.txt": "# Add your project dependencies here\n# e.g., pandas\nnumpy\n",
    "setup.py": """
from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='A short description of the project.',
    author='Your Name',
    license='MIT',
)
""",
}

# --- Content for specific files ---

# Basic GitHub Actions CI workflow
CI_YML_CONTENT = """
name: Python package

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  build:

    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.9", "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        python -m pip install -r requirements.txt
    # Add steps for linting, testing, etc. here
    # - name: Lint with flake8
    #   run: |
    #     pip install flake8
    #     # stop the build if there are Python syntax errors or undefined names
    #     flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    #     # exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
    #     flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    # - name: Test with pytest
    #   run: |
    #     pip install pytest
    #     pytest
"""


def create_structure(base_path, structure):
    """
    Recursively creates directories and files based on the provided structure.
    """
    for name, content in structure.items():
        current_path = base_path / name
        if isinstance(content, dict):
            # It's a directory with content, create it and recurse
            print(f"Creating directory: {current_path}")
            current_path.mkdir(parents=True, exist_ok=True)
            create_structure(current_path, content)
        elif isinstance(content, list):
            # It's a directory, potentially with sub-items
            print(f"Creating directory: {current_path}")
            current_path.mkdir(parents=True, exist_ok=True)
            # Create sub-items within this directory
            for item in content:
                (current_path / item).touch()
                print(f"  Creating empty file: {current_path / item}")
        else:
            # It's a file, create it and write content if provided
            print(f"Creating file: {current_path}")
            current_path.touch()
            if content:
                # Clean up leading whitespace from multiline strings
                clean_content = "\n".join(
                    [line.lstrip() for line in content.strip().split("\n")]
                )
                current_path.write_text(clean_content)


def main():
    """
    Main function to execute the script.
    """
    # The project will be created in the current working directory
    project_root = Path.cwd()
    print(f"Starting project setup in: {project_root}\n")

    # Create the main structure
    create_structure(project_root, PROJECT_STRUCTURE)

    # Special handling for files with specific content that wasn't in the main dict
    (project_root / ".github" / "workflows" / "ci.yml").write_text(
        CI_YML_CONTENT.strip()
    )

    print("\n-------------------------------------")
    print("Project structure created successfully!")
    print("-------------------------------------")


if __name__ == "__main__":
    main()
