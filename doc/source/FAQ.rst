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

Matching drivers issue here! Sam will tell you more.


**The GUI seems to be frozen after I start using it/does not work as expected**

This is normally caused by an issue with ipywidgets.
We have verified the GUI operates correctly with the 7.7.1 version,
older versions have been reported to lead to incorrect GUI behaviour.
