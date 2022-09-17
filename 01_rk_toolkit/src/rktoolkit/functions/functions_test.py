import numpy as np
from .localizers import NDMaxLocalizationFunction
from .linkers import  SimpleLinkageFunction
from ..models.graph import TreeNode, HierarchicalGraph

'''

TODO: Clean tests
Some tests are old.
Some tests are not relevant anymore
Add tests to other functions
'''
def test_NDLocalizationFunction():
    '''
    TODO: remove this. not required
    '''
    ll = NDMaxLocalizationFunction()
    x = np.linspace(0,100,100)
    y = np.linspace(101, 200, 100)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
    l = ll.predict((x,y,z))
    x1 = np.where(x==l[0])[0][0]
    y2 = np.where(y==l[1])[0][0]
    assert z.max() == z[x1, y2]

def test_simple_linkage_function():
    '''
    TODO: Update linkage function test
    '''
    sl = SimpleLinkageFunction(threshold=4)
    measures = TreeNode(parent=None, id='measures')
    hgraph = HierarchicalGraph(root=measures)
    mass = TreeNode(name="mass_measures", id='mass', parent=measures,
                    attributes={'size': 5}, value=1)
    m1 = TreeNode(name="mass1", id='m1', parent=mass, attributes={'color':'green'}, value=2)
    m2 = TreeNode(name="mass2", id='m2', parent=mass, attributes={'color': 'orange'}, value=3)

    spin = TreeNode(name="spin_measures", id='spin', parent=measures, attributes={'size': 5}, value=100)
    s1 = TreeNode(name="spin1", id='s1', parent=spin, attributes={'color':'pink'}, value=10)
    s2 = TreeNode(name="spin2", id='s2', parent=spin, attributes={'color': 'yellow'}, value=11)
    [hgraph.add_node(n) for n in [measures, mass, m1, m2, spin, s1, s2]]
    links = sl.predict(hgraph)


    valid_links = {
        's1->s2': False,
        'spin->measures': False,
        'm1->m2': False,
        'mass->measures': False

    }

    assert len(links) == 4
    for l in links:
        ll = "{}->{}".format(l.from_id, l.to_id)
        assert ll in valid_links
        valid_links[ll] = True
    for l in valid_links.items():
        assert l
