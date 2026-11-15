from common import *
import sys
for n in sys.argv[1:]:
    o=np.load(f'out/{n}_oof.npy');s=[score(y[b],o[b]) for a,b in folds];print(n,round(np.mean(s),5),round(np.std(s),5))
