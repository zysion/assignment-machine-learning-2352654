import numpy as np

def Gini_index(list_data, avr_adjacent, column):
    left = list_data[list_data[:, column] <= avr_adjacent]
    right = list_data[list_data[:, column] > avr_adjacent]

    total_sample = len(left) + len(right)
    if len(left) == 0:
        left_gini = 0
    else:
        left_gini = 1 - sum((np.sum(left[:, -1] == i) / len(left)) ** 2 for i in np.unique(left[:,-1]))
        
    if len(right) == 0:
        right_gini = 0
    else:
        right_gini = 1 - sum((np.sum(right[:,-1] == i) / len(right)) ** 2 for i in np.unique(right[:,-1]))

    return len(left) / total_sample * left_gini + len(right) / total_sample * right_gini

num_column = 562

data_file =  open(file = "co3117-ml-individual/data/UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt", mode = "r")
label_file = open(file = "co3117-ml-individual/data/UCI HAR Dataset/UCI HAR Dataset/features.txt", mode = "r")
result_file = open(file = "co3117-ml-individual/data/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt", mode = "r")
    
label_train = [line.strip() for line in label_file]
result_train = [int(line.strip()) for line in result_file]
# data_train = np.array([list(map(float, line.split())) for line in data_file])
data_train = np.empty((len(result_train), num_column), dtype=float)


label_file.close()
result_file.close()


i = 0
for line in data_file:
    data_train[i] = list(map(float, line.split())) + [int(result_train[i])]
    i += 1

data_file.close()

for c in range(0,num_column):
    sorted_data = data_train[data_train[:, c].argsort()]
    for line in range(0, len(sorted_data)-1):
        avr_adjacent = (sorted_data[line][c] + sorted_data[line+1][c])/2
        print(Gini_index(sorted_data, avr_adjacent, c))


