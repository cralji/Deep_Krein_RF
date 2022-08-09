import setuptools
  
with open("README.md", "r") as fh:
    description = fh.read()
  
setuptools.setup(
    name="Krein_Mapping",
    version="0.0.1",
    author="craljimenez",
    author_email="craljimenez@utp.edu.co",
    packages=["tensorflow 2.9.1"],
    description="A sample test package",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/cralji/Deep_Krein_RF.git",
    license='MIT',
    python_requires='>=3.8',
    install_requires=[]
)