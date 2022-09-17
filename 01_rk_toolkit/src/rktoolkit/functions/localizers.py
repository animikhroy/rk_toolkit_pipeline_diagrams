'''
Localization functions

Must implement the LocalizationFunction interface
'''
import numpy as np
from ..models.functions import LocalizationFunction
from typing import Optional
from pydantic import PrivateAttr

class NDMaxLocalizationFunction(LocalizationFunction):

    def __init__(self):
        super().__init__()

    def predict(self, X):
        '''
        given a matrix NxD. Localizes around the max of the matrix
        '''
        pos = np.argmax(X[2])
        i1=int(X[2].argmax()/len(X[2])),
        i2 = int(X[2].argmax() % len(X[2]))
        return X[0][i1], X[1][i2]

class IterableLocalizationFunction(LocalizationFunction):

    _iterateX: bool = PrivateAttr()
    _iterateY: bool = PrivateAttr()
    _iterateZ: bool = PrivateAttr()

    _xcount: int = PrivateAttr()
    _ycount: int = PrivateAttr()
    _zcount: int = PrivateAttr()

    def __init__(self, iterateX=True, iterateY=True, iterateZ=True):
        super().__init__()
        self._iterateX = iterateX
        self._iterateY = iterateY
        self._iterateZ = iterateZ
        self._xcount = -1
        self._ycount = -1
        self._zcount = -1

    def localize(self, X):
        '''
        given a matrix NxD. Localizes around the max of the matrix
        '''
        if self._iterateX:
            self._xcount += 1
        if self._iterateY:
            self._ycount += 1
        if self._iterateZ:
            self._zcount += 1

        return self._xcount, self._ycount, self._zcount
