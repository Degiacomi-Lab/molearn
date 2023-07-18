##########################
Frequently Asked Questions
##########################


I get an IndexError when I try loading a multiPDB
-------------------------------------------------

This is likely an error thrown by MDAnalysis. Typically this happens when
attempting to load a multiPDB file saved with software like VMD, which uses a
different syntax to indicate the end of a conformer in the file. A way to get
around this, is to re-save the file in a format MDAnalysis can parse, e.g., by
loading and re-saving the file via biobox.

.. code-block::

    import biobox as bb
    M = bb.Molecule(filename)
    M.write_pdb(newfilename)


I cannot install openmmtorchplugin
----------------------------------

openmmtorchplugin depends on conda-forge builds of pyTorch and OpenMM.
Due to this dependency, Windows cannot be supported.
Installation can be carried out via terminal with conda-forge:

.. code::

    conda install -c conda-forge openmmtorchplugin

The following Python versions are supported: 3.8, 3.9, 3.10, 3.11.
If you are running into any issue, attempt a fresh install in a new conda
environment:

.. code:: 

    conda create --name test_env python=3.10
    conda install -c conda-forge openmmtorchplugin molearn

openmmtorchplugin is built with cuda_compiler_version=11.2 in conda-forge CI tools.
This has been successfully tested on Ubuntu machines running with the driver
version 525.105.17 (see nvidia-smi output).

The Nvidia website tabulates minimum driver versions required and version compatibility:
`NVIDIA CUDA Toolkit Minimum driver versions <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>`_


The GUI freezes when I use it/does not work as expected
-------------------------------------------------------

This is usually caused by an issue with packages handling communications between the GUI and Jupyter, `see here <https://discourse.jupyter.org/t/jupyter-notebook-zmq-message-arrived-on-closed-channel-error/17869/27>`_. 
Currently, a workaround is to use older versions of `tornado`.
In Python 3.10, the following packages have been observed to yield correct behaviour:

.. code:: 

    ipywidgets=8.0.7
    nglview=3.0.6
    tornado=6.1
