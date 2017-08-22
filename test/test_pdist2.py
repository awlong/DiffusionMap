import PyDMap
import numpy as np
A=np.asarray([[ 1., -1.,  0., -1.,  0.,  0.,  0.,  0.],
             [-1.,  1., -1.,  1.,  0.,  1.,  0.,  0.],
             [ 0., -1.,  1.,  0.,  0., -1.,  0.,  0.],
             [-1.,  1.,  0.,  1., -1.,  0.,  1.,  0.],
             [ 0.,  0.,  0., -1.,  1.,  0., -1.,  0.],
             [ 0.,  0., -1.,  0.,  0.,  1.,  0., -1.],
             [ 0.,  0.,  0.,  1., -1.,  0.,  1., -1.],
             [ 0.,  0.,  0.,  0.,  0., -1., -1.,  1.]])
B=np.asarray([[ 1.,  1., -1.,  0.],
             [ 1.,  1., -1.,  0.],
             [-1., -1.,  1., -1.],
             [ 0.,  0., -1.,  1.]])

ISO = PyDMap.MatchHeuristic.ISORANK
SIG_ISO = PyDMap.MatchHeuristic.SIGNED_ISORANK

print("PyDMap.dist(A,B,ISO)=", (PyDMap.dist(A,B,ISO)))
print("PyDMap.dist(A,B,SIG_ISO)=", (PyDMap.dist(A,B,SIG_ISO)))
print("PyDMap.pdist2(A,B,ISO)=", (PyDMap.pdist2(A,B)))
print("PyDMap.pdist2([A],B,ISO)=", (PyDMap.pdist2([A],B)))
print("PyDMap.pdist2(A,[B],ISO)=", (PyDMap.pdist2(A,[B])))
print("PyDMap.pdist2([A],[B],ISO)=", (PyDMap.pdist2([A],[B])))
print("PyDMap.pdist2([A],[A,B],ISO)=", (PyDMap.pdist2([A],[A,B])))
print("PyDMap.pdist2([A],[A,B],SIG_ISO)=", (PyDMap.pdist2([A],[A,B],SIG_ISO)))
print("PyDMap.pdist2([B],[A,B],ISO)=", (PyDMap.pdist2([B],[A,B])))
print("PyDMap.pdist2([B],[A,B],SIG_ISO)=", (PyDMap.pdist2([B],[A,B],SIG_ISO)))
print("PyDMap.pdist2([B,A],[A,B],SIG_ISO)=", (PyDMap.pdist2([B,A],[A,B],SIG_ISO)))

