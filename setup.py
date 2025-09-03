from setuptools import setup, find_packages

setup(
    name="eye_trance",
    version="1.0.0",
    description="Eye Trance Art Project",
    url="https://github.com/michi-d/eye_trance",
    author="Nadia / Lana / Michi",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26",
        "matplotlib>=3.10",
        "pillow>=11.2",
        "glfw",
        "numpy",
    ],
)
