# RK Toolkit

Standard package and toolkit for RK Diagrams.

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

### Demos
Check the notebooks sections for more details.
