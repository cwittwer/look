from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="python-look",
    version="0.1.6",
	license="MIT",
    author="Carson Wittwer",
    author_email="wittwer.carson@googlemail.com",
    description="Simple wrapper code to run inference on images using the model and code from the Looking Repo from VITA-EPFL.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cwittwer/look",
    packages=find_packages(exclude=("tests", "requirements.txt",)),
	include_package_data=True,
	install_requires=[
    "tqdm",
    "numpy",
    "opencv_python==4.7.0.72",
    "seaborn",
    "matplotlib",
    "torch",
    "torchvision",
    "openpifpaf==0.13.11",
    "scikit_image==0.19.3",
    "Pillow",
    "scikit_learn"
	],
    classifiers=[
      "Development Status :: 4 - Beta",
		  "Intended Audience :: Developers",
      "License :: OSI Approved :: MIT License",
      "Operating System :: OS Independent",
      "Programming Language :: Python :: 3.6",
		  "Programming Language :: Python :: 3.7",
		  "Programming Language :: Python :: 3.8",
		  "Programming Language :: Python :: 3.9"
    ],
    python_requires='>=3.6, <3.10'
)
