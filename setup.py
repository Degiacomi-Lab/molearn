import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="molearn",
    version="2.0.1",
    author="S. C. Musson",
    long_description = long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where='src',),
    package_data={'molearn':['parameters/*']},
    install_requires = ["numpy", "pytorch", "biobox", "openmmtorchplugin"],
)
