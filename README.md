# ACL
Adapted Caldeira-Legget Model

[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](https://unitary.fund)

# Overview

Quantum decoherence is a crucial concept in quantum mechanics, describing the process by which a quantum system loses coherence due to interactions with its environment. It is important to understand on a practical level for quantum technologies as well as a foundations level with its importance to the the measurement problem. The ACL (Adapted Caldeira-Legget Model) model is a very useful toy model for studying the phenomena from a theoretical and numerical framework. It was introduced by Andreas Albrecht in https://arxiv.org/abs/2105.14040. It is a simplification of the CL model, built in finite Hilbert space making it practical for numerical study without non-unitary approximations.

This repository offers a computational implementation of the ACL model using Qutip. 

This repository offers an open-source quantum decoherence model implemented using Qutip in the form of a package (in the near future). Alongside the simulation, it provides a comprehensive set of tools and utilities for analyzing and visualizing decoherence processes, with the goal of being a one-stop shop for people wanting to generate numerics using the ACL model.

The goal is to make this as useful a package as possible for research. Useful for easily verifying one's intuition and useful for generating numerical results for a paper. The package will be easily adaptable to one's specific requierments, in the sense that it aims to provide a foundation on which someone can come and add their own functions to study decoherence.

# Features

(DONE) ACL Model Simulation: Generate unitary evolution of an interacting system + environment depending on relevant ACL parameters.

(IN PROGRESS) Analyze the effects of decoherence: visualize apparent collapse, capture Schmidt states convergence to Pointer states, see splitting of total energy eigenspaces, VN entropy growths, ...

(IN PROGRESS) Visualization Tools: See the dynamics of your system via generation of gifs.

Modular Design: The codebase is designed with modularity in mind, allowing easy extension and customization for specific research needs.

(COMING) Documentation and Examples: Comprehensive documentation and example scripts are provided to facilitate usage and understanding of the model and its capabilities. Tutorial Jupyter notebooks.

# How to clone and use this model will be detailed once it is usable.

# Acknowledgments

Huge thanks to the Unitary fund for funding the work done here.

For any inquiries or feedback regarding this project, feel free to contact marin.girard(at)outlook.com.
