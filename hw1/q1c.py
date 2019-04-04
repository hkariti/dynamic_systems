import sys
import numpy as np

cached_calcs = None

def init(N, X):
    global cached_calcs
    cached_calcs = -np.ones((N, X))

def psi(N, X):
    if N > X:
        return 0
    if N == 1:
        return 1
    global cached_calcs
    result = 0
    for x in range(1, X-(N-1)+1):
        rec_N = N - 1
        rec_X = X - x
        cached = cached_calcs[rec_N-1, rec_X-1]
        if cached != -1:
            psi_recurse = cached
        else:
            psi_recurse = psi(rec_N, rec_X)
            cached_calcs[rec_N-1, rec_X-1] = psi_recurse
        result = result + psi_recurse
    return result

N = int(sys.argv[1])
X = int(sys.argv[2])

print(f"Calculating psi_{N}({X})")
init(N, X)
print("Result:", psi(N, X))
