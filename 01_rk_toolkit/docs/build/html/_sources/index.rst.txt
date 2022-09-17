.. rktoolkit documentation master file, created by
   sphinx-quickstart on Sun May 16 14:42:05 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

RK-Toolkit Documentation
=====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Readme File
===========

Example Pipeline
======================

.. code-block:: python

   example_pipeline = RKPipeline(preprocess_nodes=[MinMaxNormalizerNode()],
                                 localization_algorithm=MaxLocalizer(),
                                 hierarchical_embedding_nodes= [
                                    {
                                       "HFeatureExtractor1": HierarchicalFeatureExtractor1()
                                    }
                                 ],
                                 filter_functions=[
                                    {
                                       "HFeatureExtractor1" :
                                       {
                                             'range_measure': StaticFilter(min=.2, max=.8),
                                             'max_measure': StaticFilter(min=0, max=1)
                                       }
                                    }
                                 ], # question: how to define which limits for which measure. Each filter and linkage has to be BY CLUSTER
                                 linkage_function=SimpleLinkage(threshold=.8))
                                 example_pipeline.build()
   example_pipeline.fit(X)
   rk_model = example_pipeline.transform(X)
   rk_models.append(rk_model)

   visualizer = RKModelVisualizer(method="circular")
   visualizer.build(rk_models) # build requires a list of rk_models
   visualizer.show()
