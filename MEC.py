import numpy as np


def hamming_distance(read, haplo):
    return sum((haplo - read)[np.where(read != 0)] != 0)


# calculate MEC
def calculate_MEC(SNVmatrix, Recovered_Haplo):
    res = 0

    for i in range(len(SNVmatrix)):
        dis = [hamming_distance(SNVmatrix[i, :], Recovered_Haplo[j, :]) for j in range(len(Recovered_Haplo))]
        res += min(dis)

    return res
