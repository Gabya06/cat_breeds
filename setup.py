from setuptools import setup, find_packages

setup(
    name="cat_project",  # Name of the package
    version="0.1",  # Version of the package
    packages=find_packages(),  # Automatically find all packages
    install_requires=[  # List of dependencies
        "requests",
        "chromadb==1.0.3",
        "google-genai==1.7.0",
        "matplotlib>=3.8.0",
        "Pillow>=9.4.0",
        "openai==0.27.2",
        "transformers==4.45.2",
        "torch==2.6.0",
        "numpy>=1.26.0",
    ],
    entry_points={  # Optionally add CLI commands if needed
        # "console_scripts": [
        #     "cat_project=cat_project.cli:main",  # Example if you want a CLI
        # ],
    },
    include_package_data=True,  # Include non-Python files (like images) in the package
)
