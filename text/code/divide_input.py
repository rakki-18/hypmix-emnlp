import numpy as np
import torch

def group(input_a, target_a, length_a):
    length = input_a.shape[0]

    # Find max_dist pair and make the pair as two heads of the 2 groups
    max_dist_pair = [0,0]
    max_dist = 0
    for i in range(length):
        for j in range(length):
            dist = np.linalg.norm(input_a[i] - input_a[j])
            if dist > max_dist:
                max_dist = dist
                max_dist_pair = [i,j]

    dist = torch.Tensor(length,2)
    # Find similarity between one head of group and all other points
    for i in range(length):
        dist[i][0] = np.linalg.norm(input_a[i] - input_a[max_dist_pair[0]])
        dist[i][1] = i
    print(dist)
    dist_sorted = dist[dist[:, 0].sort()[1]]
    print("after sorting")
    print(dist_sorted)

    # Based on the similarity scores sorted,
    # Put the first half on one group
    # And the other half on the second group
    first_group = dist_sorted[0:int(length/2),1].long()
    second_group = dist_sorted[int(length/2):length,1].long()
    print(first_group)

    # Return the two divided inputs, on which the mix operation will be performed later
    return input_a[first_group], input_a[second_group], target_a[first_group], target_a[second_group], length_a[first_group], length_a[second_group]

