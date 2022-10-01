# Table of Contents

To use:

```
git clone git@github.com:andorsk/rk_toolkit.git
git submodule update --init --recursive
```
### 01_rk_toolkit

A generalized implementation of the  [A Novel Approach to Topological Graph Theory with R-K Diagrams and Gravitational Wave Analysis](https://arxiv.org/abs/2201.06923) written by **Animikh Roy** and **Andor Kesselman**. The **“R-K Toolkit”** is a generalized code package implemented to aid users in the process of building “R-K Diagrams” and “R-K Models” via the “R-K Pipeline”. The **R-K** terminology originates form the last names of the co-authors **Roy & Kesselman** respectively.

The R-K Toolkit provides a computational framework to build a functional component called the R-K Pipeline, which can be used to transform any NxM Tensor with 3 or more independent physical/ontological variables, into a R-K Model. An “R-K Model ‘’ is then processed with domain appropriate “Range-Filters” and “Leaf-Linkers” to render unique topological signatures and graph visualizations that are known as R-K Diagrams.It's essentially serves as a utility package ( like scikit-learn ) specifically built to support computational transfroms and objects in
the R-K Pipeline. A set of sample R-K Diagrams pertaining to Compact Binary Coalescence (CBC) merger events rendered via the R-K Toolkit and Pipeline using LIGO (https://www.gw-openscience.org/) data, has been demonstrated below for reference puroses:

![R-K Model](https://user-images.githubusercontent.com/55942592/193401780-87de09e8-d182-4bf9-8604-c1a5938738df.gif)


### 02_notebooks

Implementation of notebooks used in the paper. Specifically, two notebooks.

* Tableau Sales TDA w/ RK Diagrams
* Ligo PBH Classification 

### 03_rk-visualizer

RK Visualizer is a tool that was made as a prototype to demonstrate the RK
Diagrams. It does not use any of the core toolkit features, as was made as a
seperate initiative.


### 04_publications 

Publication documents for arxiv and peer review.

### Tree
``` sh
.
├── 01_rk_toolkit
│   ├── src
│   │   └── rktoolkit
│   │       ├── functions
│   │       ├── io
│   │       ├── models
│   │       ├── preprocess
│   │       └── visualizers
│   └── tests
├── 02_notebooks
│   ├── rk_general_applications
│   ├── rk_gw_mma
│   └── src
├── 03_rk-visualizer
│   └── rkmodel
└── 04_publications
    ├── arxiv
    │   └── A\ Novel\ Approach\ to\ Topological\ Graph\ Theory\ with\ R-K\ Topohedrons
    │       ├── bib
    │       ├── images
    │       ├── notes
    │       ├── scripts
    │       └── sections
    ├── physical_reviewD
    │   └── images
    └── physical_reviewX
        └── images

```
