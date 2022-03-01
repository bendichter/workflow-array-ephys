# DataJoint Workflow - Array Electrophysiology

Workflow for extracellular array electrophysiology data acquired with a polytrode probe (e.g. 
Neuropixels, Neuralynx) using the `SpikeGLX` or `OpenEphys` acquisition software and processed 
with MATLAB- or python-based `Kilosort` spike sorting software.

A complete electrophysiology workflow can be built using the DataJoint Elements.
+ [element-lab](https://github.com/datajoint/element-lab)
+ [element-animal](https://github.com/datajoint/element-animal)
+ [element-session](https://github.com/datajoint/element-session)
+ [element-array-ephys](https://github.com/datajoint/element-array-ephys)

This repository provides demonstrations for:
1. Set up a workflow using DataJoint Elements (see 
[workflow_array_ephys/pipeline.py](workflow_array_ephys/pipeline.py))
2. Ingestion of data/metadata based on a predefined file structure, file naming 
convention, and directory lookup methods (see 
[workflow_array_ephys/paths.py](workflow_array_ephys/paths.py)).
3. Ingestion of clustering results.

## Workflow architecture

The electrophysiology workflow presented here uses components from 4 DataJoint 
Elements (`element-lab`, `element-animal`, `element-session`, 
`element-array-ephys`) assembled together to form a fully functional workflow.

### element-lab

![element-lab](
https://github.com/datajoint/element-lab/raw/main/images/element_lab_diagram.svg)

### element-animal

![element-animal](
https://github.com/datajoint/element-animal/blob/main/images/subject_diagram.svg)

### assembled with element-array-ephys

![element-array-ephys](images/attached_array_ephys_element.svg)

## Installation instructions

+ The installation instructions can be found at [datajoint-elements/install.md](
     https://github.com/datajoint/datajoint-elements/blob/main/install.md).

## Interacting with the DataJoint workflow

+ Please refer to the following workflow-specific 
[Jupyter notebooks](/notebooks) for an in-depth explanation of how to run the 
workflow ([03-process.ipynb](notebooks/03-process.ipynb)) and explore the data 
([05-explore.ipynb](notebooks/05-explore.ipynb)).