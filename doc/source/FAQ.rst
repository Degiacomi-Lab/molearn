Frequently Asked Questions
--------------------------

**I get an IndexError when I try loading a multiPDB**

This is likely an error thrown by MDAnalyis. Typically this happens when
attempting to load a multiPDB file saved with software like VMD, which uses a
different syntax to indicate the end of a conformer in the file. A way to get
around this, is to resave the file in a format MDAnalysis understands, e.g. by
loading and re-saving the file via biobox.

.. code-block::

    import biobox as bb
    M = bb.Molecule(filename)
    M.write_pdb(newfilename)


**I cannot install openmmtorchplugin**

openmmtorchplugin must be install via terminal with conda-forge:

.. code::

    conda install -c conda-forge openmmtorchplugin

The following python versions are supported: 3.8, 3.9, 3.10, 3.11
If you are running into issues please attempt a fresh install in a new conda environment:

.. code:: 

    conda create --name test_env python=3.10
    conda install -c conda-forge openmmtorchplugin molearn

openmmtorchplugin depends on conda-forge builds of pytorch and openmm. Due to this dependency Windows can not be supported. It is built with cuda_compiler_version = 11.2 in conda forge CI tools.
This has been successfully tested on Ubuntu machines running with a driver version of 525.105.17 (see nvidia-smi output).

The nvidia website tabulats minimum driver versions required and version compatibility. `NVIDIA CUDA Toolkit Minimum driver versions <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>`_



**The GUI seems to be frozen after I start using it/does not work as expected**

This is normally caused by an issue with ipywidgets.
We have verified the GUI operates correctly with the 7.7.1 version,
older versions have been reported to lead to incorrect GUI behaviour.
