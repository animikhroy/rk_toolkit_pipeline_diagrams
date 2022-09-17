# RK-Diagrams and Pipeline

The following repo is a generalized implementation of the 
[A novel approach ot TDA: RK Diagrams Paper]() written by Andor Kesselman 
and Animikh Roy.

The core packages of this can be extended to many use csaes. Check out the 
LIGO DK-Diagram Pipeline for a specific implementation done on the LIGO dataset

#### Installation
``` sh
python -m pip install .
```

#### Using a module

**TODO: FIX**
To use a module:
``` sh
from rktoolkit.visualizers.networkx.dendrogram import hierarchy_pos
from rktoolkit.pipeline import RKPipeline

# make filters and get structural graph
pipeline = RKPipeline(filter_map=filters, linkage_map=linkers, structural_graph=g) 

# get the base transform
base = hft.transform(event)
rkm = pipeline.transform(base, is_base=False)
visualize_rkmodel(rkm)
```

##### Running Tests
``` sh
pytest -m .
```

### 
Check the notebooks sections for more details.
