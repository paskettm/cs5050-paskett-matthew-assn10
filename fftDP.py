import numpy as np
from math import *
import cmath
import time
import matplotlib.pyplot as plt


def fftR(p, v, n, depth=0):
    # print("%s%s" % (" |"*depth+"in =", str(p)))
    # base case
    if n == 1:
        # print("%s%s" % (" |"*depth+"out=", str(p)))
        return p
    # split into even and odd
    eve = [p[i] for i in range(0, n, 2)]
    odd = [p[i] for i in range(1, n, 2)]
    # square the v values
    v2 = [v[i]*v[i] for i in range(0, n//2)]
    # solve the two sub problems
    eveS = fftR(eve, v2, n//2, depth+3)
    oddS = fftR(odd, v2, n//2, depth+3)
    # construct the solution
    solution = ([eveS[i] + v[i]*oddS[i] for i in range(0, n//2)] +
                [eveS[i] - v[i]*oddS[i] for i in range(0, n//2)])
    # print("%s%s" % (" |"*depth+"out=", str(solution)))

    return solution


def getV(n, sign=1):
    return [complex(cos(2*pi*i/n), sign * sin(2*pi*i/n)) for i in range(0, n)]


def rbs(num, places):
    numStr = "{:0" + str(places) + "b}"
    result = int(numStr.format(num)[::-1], 2)
    return result


# p = array to be used
# v = array of omega values to be used
def fftDP(p, v, n):
    log2n = int(log2(n))
    # Initialize cache
    cache = np.full((log2n + 1, n), np.complex)
    # Fill in base cases after reverse bit shuffle
    for i in range(len(p) - 1):
        p[i] = rbs(p[i], log2n)
    cache[0] = p
    # Begin loops to fill in cache
    for i in range(1, int(log2n) + 1):
        size = int(pow(2, i))
        depth = len(cache) - 1 - i
        for j in range(0, n, size):
            for k in range(0, size // 2):
                cache[i, j+k] = cache[i-1, j+k] + v[k*2**depth] * cache[i-1, (j+size//2)+k]
                cache[i, (j+size//2)+k] = cache[i-1, j+k] - v[k*2**depth] * cache[i-1, (j+size//2)+k]

    return cache[log2n]


if __name__ == "__main__":
    powers = []
    timeDP = []
    timeR = []
    maxPower = 16
    for i in range(6, maxPower):
        p = np.random.randint(0, 7, 2**i)
        n = len(p)
        v = getV(n)
        powers.append(i)
        # Time for DP algo
        start = time.time()
        ansDP = fftDP(p, v, n)
        timeDP.append(time.time() - start)
        # Time for recursive function
        start = time.time()
        ansR = fftR(p, v, n)
        timeR.append(time.time() - start)

    # Comparing answers
    ansNum = 1
    print(f"Recursive answer[{ansNum}]:\n{ansR[ansNum]}\nDP answer[{ansNum}]:\n{ansDP[ansNum]}\n")

    plt.plot(powers, timeDP, label="D.P. FFT")
    plt.plot(powers, timeR, label="Recursive FFT")
    plt.yscale('log', basey=2)
    plt.title("FFT (DP/Recursion) Runtime v Problem Size")
    plt.xlabel("Problem Size (log 2)")
    plt.ylabel("Time (log 2)")
    plt.legend()
    plt.savefig(f"DPvRec_FFT_n={maxPower}.png")
    plt.show()
    print(f"D.P. Time\n{timeDP}\nRecursion Times\n{timeR}")
