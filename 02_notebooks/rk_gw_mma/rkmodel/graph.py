#!/usr/bin/env python

import uuid

class Node:

    def __init__(self, label, id=None, value=None, attributes=None):
        attributes = {} if attributes is None else attributes
        self.label = label
        self._id = id if id is not None else uuid.uuid4()
        self.value = value
        self.attributes = attributes

    def get_id(self):
        return self._id

    def serialize(self):
        return json.dumps({
            'label': self.label,
            'id': self._id,
            'value': self.value,
            'attributes': self.attributes
        })

    def to_nx_node(self):
        att = self.attributes
        att['value'] = self.value
        return (self._id, att)

class Edge:

    def __init__(self, from_id, to_id, label="", weight=1, type="directed"):
        self.from_id = from_id
        self.to_id = to_id
        self.weight = weight
        self.type = type
        self.label = label

    def serialize(self):
        return {
            'from': self.from_id,
            'to': self.to_id,
            'weight': self.weight,
            'type': self.type,
            'label': self.label
        }


    def to_nx_edge(self):
        return (self.from_id, self.to_id, {'weight': self.weigth, 'type': self.type, 'label': self.label})
