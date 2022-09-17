import abc
from collections.abc import Sequence
from typing import List
from ..models.graph import RKModel

class RKModelWriter(metaclass=abc.ABCMeta):
    '''
    Datastore Interface

    Writes a rk-model.
    '''
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, 'write') and callable(subclass.write)

    def write(self, model: RKModel) -> bool: # returns whether success or failure on write
        pass

class RKModelReader(metaclass=abc.ABCMeta):
    '''
    RK Model Reader Interface
    Must implement the following interfaces
    '''
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, 'read') and callable(subclass.read) and \
            hasattr(subclass, 'next') and callable(subclass.next) and  \
            hasattr(subclass, 'close') and callable(subclass.close) and  \
            hasattr(subclass, 'readAll') and callable(subclass.readAll)

    def read(self) -> RKModel:
        pass

    def next(self) -> RKModel:
        pass

    def close(self):
        pass

    def readAll(self) -> List[RKModel]:
        pass
