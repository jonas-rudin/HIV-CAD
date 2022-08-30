import numpy as np


def get_reads_from_sam(path):
    with open(path, 'r') as file:
        all_lines = file.readlines()
        for line in all_lines:
            list_of_line = line.split()
            print(list_of_line[0])
            print(list_of_line[1])


def hamming_distance(read, haplo):
    return sum((haplo - read)[np.where(read != 0)] != 0)


# calculate MEC
def calculate_MEC(SNVmatrix, Recovered_Haplo):
    res = 0

    for i in range(len(SNVmatrix)):
        dis = [hamming_distance(SNVmatrix[i, :], Recovered_Haplo[j, :]) for j in range(len(Recovered_Haplo))]
        res += min(dis)

    return res
