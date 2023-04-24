---
title: 'Molearn: A Python package streamlining the design of generative models of biomolecular dynamics'
tags:
  - Python
  - machine learning
  - molecular dynamics
  - proteins
  
authors:
  - name: Samuel C. Musson
    orcid: 0000-0002-2189-554X
    equal-contrib: false
    affiliation: 1
  - name: Matteo T. Degiacomi
    orcid: 0000-0003-4672-471X
    equal-contrib: false
    corresponding: true
    affiliation: 1
affiliations:
 - name: Department of Physics, Durham University, UK
   index: 1
date: 24 August 2023
bibliography: paper.bib

---

# Summary

We present `molearn`, a Python package facilitating the implementation of machine 
learning models dedicated to the generation of biomolecular conformations from
example conformers obtained via experiment or molecular simulation.


# Statement of need

Most biological mechanisms directly involve biopolymers such as proteins and DNA. The specific task each of these molecules carries out is linked to their
three-dimensional shape, enabling them to bind to designated binding partners such as small molecules, ions, or other biopolymers. Crucially though,
biomolecules are flexible and so are continuously jostling and reconfiguringdue to Brownian motion. Thus, their function emerges from characteristic
conformational dynamics. Characterising structure and dynamics of biomolecules at the atomic level provides us with a fundamental understanding of the mechanisms underpinning life and is the first step in numerous technological applications. Progress in these areas has been spearheaded by the development of a diverse palette of dedicated experimental techniques. Unfortunately, none is singlehandedly capable of routinely reporting on the full fine structure of biomolecular conformational spaces. Most low-resolution experimental techniques (e.g., SAXS, FRET) report on ensemble averages, and those reporting on atomic structure (e.g., XRD, NMR, EM) more easily yield one or few snapshots of low-energy states. Our understanding of life at the molecular level is
inherently biased by the techniques we adopt to observe it [@Marsh2015]. Most experimental techniques are ill-suited to provide atomic-level insight
into the full conformational space of a protein. Resolving a protein at a high-energy transition state, can be seen as a technical achievement.

Molecular dynamics (MD) simulations can give atomistic insights into the conformational landscape of biomolecules, complementing and extending data
gathered experimentally. MD simulations produce samples of molecular conformational spaces by iteratively generating new conformers based on an
initial, known atomic arrangement and physical models of atomic interactions. While MD enables obtaining key insight into biomolecular function, it is not a
silver bullet: exhaustive sampling of key biological processes such as folding or spontaneous binding with partners usually lay beyond what can be routinely
observed. Generative Neural Networks (GNNs) have been shown to be effective predictors of proteins 3D structure from their sequence [@Jumper2021; @Baek2021]. Several efforts have also demonstrated that a neural network trained with MD conformers can learn a meaningful dimensionality reduction of the data, usable for reaction coordinate definition [@frassek2021extended; @chen2018collective], or driving conformational space sampling [@noe2019boltzmann; @sidky2020molecular; @mehdi2022accelerating]. In this context, we have previously presented GNNs capable of generating protein conformations based on small pools of examples produced from MD [@Degiacomi2019; @Ramaswamy2021].

The issue is that developing an ML model to study biomolecular dynamics is a lengthy process. This requires setting up means of transforming conformational space data into tensor data submittable to a model, as well as assessing models quality (e.g., in terms of their energy or according to structural descriptors). Here we present `molearn`, a Python framework facilitating the implementation of generative neural networks learning protein conformational spaces from examples obtained via experiments or MD simulations.


# Package Description

 Classes available provide support for the following tasks:
-	*Data loading*. We provide methods to parse molecular conformers and convert them in a pyTorch [@paszke2019pytorch] tensor format suitable for training.
-	*Model design*. Molearn comes with a range of pre-implemented models, ready to be trained on any desired datasets.
-	*Loss function definition*. While the classical loss function in a generative model typically builds upon a mean square error between input and output, here we provide the capability of directly interacting with the OpenMM molecular dynamics engine [@eastman2017openmm]. Specifically, we have implemented means of pushing data in pyTorch tensor format directly into OpenMM backend. This enables quickly evaluating the energy of a generated model according to any force field accepted by OpenMM. This also enables directly running MD simulations with generated conformers while the model trains.
-	*Model analysis*. Once a model is trained, it is important to gather metrics defining the quality of its models. We provide tools to quickly quantify the DOPE score [@shen2006statistical] and Ramachandran plot of generated conformers, as well as the RMSD between generated models and known conformers We also provide a graphical user interface, enabling the visualisation of neural network latent space via plotly [@plotly]. The GUI also enables generating and visualising predicted interpolations with a 3D view supported by via nglview [@nguyen2018nglview].

![`molearn` analysis tools include a graphical user interface, enabling the on-demand generation of protein conformations. The panel on the left controls how the neural network latent space is presented, the central panel is a plotly interactive panel, the panel on the right is a representation of an interpolation through the latent space supported by nglview\label{fig:gui}](gui.png)


# Usage

We advise training the model on a GPU infrastructure. `molearn` comes with a series of tutorial examples, also usable as-is to train on MD dataset.
Full package API is available. A tutorial on neural network analysis is also available [cite]. This also includes a GUI to directly interact with a trained neural network \autoref{fig:gui}.

Results obtanable via `molearn` are exemplified in [@Ramaswamy2021]. Here, we designed and trained a 1D convolutional autoencoder against protein molecular dynamics simulation data. The GNN was trained via a loss function directing the neural network to both faithfully reconstruct
training data, and produce low-energy interpolations between them, whereby energy is assessed according to an MD force field.


# Acknowledgements

We thank the first users of `molearn`, Lucy Vost, Louis Sayer, Cameron Stewart, and Ryan Zhu, for their feedback.
Matteo T. Degiacomi acknowledges support from the Engineering and Physical Sciences Research Council (EP/P016499/1).


# References
