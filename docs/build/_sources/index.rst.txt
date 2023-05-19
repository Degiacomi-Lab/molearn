.. molearn documentation master file, created by
   sphinx-quickstart on Sat Jan 16 17:07:10 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to molearn's documentation!
===================================

molearn is a Python package streamlining the implementation of machine learning
models dedicated to the generation of protein conformations from example data
obtained via experiment or molecular simulation.

molearn provides tools to load protein conformational ensemble data, build
a model handling this data, define a loss function for this model to train against,
define a training protocol, and analyse the resulting trained model.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   data
   loss_functions
   models
   trainers
   scoring
   analysis
   FAQ
   

Please see molearn's `Github page <https://github.com/Degiacomi-Lab/molearn>`_ 
for installation instructions and examples scripts.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
