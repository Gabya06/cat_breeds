from setuptools import setup, find_packages

setup(
    name="cat_project",  # Name of the package
    version="0.1",  # Version of the package
    packages=find_packages(),  # Automatically find all packages
    install_requires=[  # List of dependencies
        "requests",
        "chromadb==1.0.3",
        "google-genai==1.7.0",
        "matplotlib",
        "Pillow",
        "openai",
        "clip",
        "transformers",
        "torch",  # Add any other dependencies you need
    ],
    entry_points={  # Optionally add CLI commands if needed
        "console_scripts": [
            "cat_project=cat_project.cli:main",  # Example if you want a CLI
        ],
    },
    include_package_data=True,  # Include non-Python files (like images) in the package
)
