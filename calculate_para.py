import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import re
from numpy import linalg as LA

num = '013'
topic_order = []

model_dir = '/Users/Martin/Desktop/Fall2016/plda/tmp/model_'+num+'.txt'
inf_dir = '/Users/Martin/Desktop/Fall2016/plda/tmp/inf_'+num+'.txt'
corpus_dir = '/Users/Martin/Desktop/Fall2016/plda/testdata/corpus_'+num+'.txt'
alpha = 10
beta = 0.01
n_topics = 4

for i in range(n_topics):
     topic_order.append(i)

def od_to_word(o, d):
    return str(o) + '_to_' + str(d)


def normailize_row(matrix):
    res = np.zeros(matrix.shape)
    sums = np.sum(matrix, axis=1)
    for i in range(matrix.shape[0]):
        res[i] = matrix[i]/sums[i]
    return res

def normailize_column(matrix):
    res = np.zeros(matrix.shape)
    sums = np.sum(matrix, axis=0)
    for i in range(matrix.shape[1]):
        res[:,i] = matrix[:,i]/sums[i]
    return res

def kl_div_1(A, B):
    size = A.shape[0]
    div = 0
    for i in range(size):
        for j in range(size):
            if A[i,j] == 0:
                div += 0
            elif B[i,j] == 0:
                div += A[i, j] * np.log(A[i, j])
            else:
                div += A[i, j] * np.log(A[i, j]/B[i, j])
    return div

def kl_div(A, B):
    M = (A + B) /2
    return (kl_div_1(A, M) + kl_div_1(B, M)) / 2


# corpus file
corpus_file = []
with open(corpus_dir, 'rt') as f:
    for line in f:
        line = re.split(('\n|\t| '), line)
        line.remove('')
        line.pop()
        corpus_file.append(line)
D = len(corpus_file)
inf_file_temp = []
with open(inf_dir, 'rt') as f:
    for line in f:
        line = line.split()
        inf_file_temp.append(line)

count_c = 0
count_i = 0

# inference file
inf_file = []
while count_c < len(corpus_file):
    if len(corpus_file[count_c]) > 0:
        inf_file.append(inf_file_temp[count_i])
        count_c += 1
        count_i += 1
    else:
        inf_file.append([0] * n_topics)
        count_c += 1

topic_order[0]=np.argmax(np.array(inf_file[31], dtype=float))
topic_order[1]=np.argmax(np.array(inf_file[41], dtype=float))
topic_order[2]=np.argmax(np.array(inf_file[12], dtype=float))
topic_order[3]=np.argmax(np.array(inf_file[17], dtype=float))



# print(topic_order)
# model file
model_dic = {}
model_file = []

with open(model_dir, 'rt') as f:
    count = 0
    for line in f:
        line = line.split()
        model_dic[line[0]] = count
        model_file.append(line)
        count += 1
W = len(model_file)



overall = np.zeros(len(model_file))
overall_sum = 0
for i in range(len(model_file)):
    overall[i] = np.sum(np.array(model_file[i][1:], dtype=float))
    overall_sum += overall[i]
overall = overall/overall_sum



phi = np.zeros((W, n_topics))
theta = np.zeros((D, n_topics))
# calculate n_j
n_j = [0] * n_topics
for t in range(n_topics):
    for w in range(W):
        n_j[t] += int(float(model_file[w][t + 1]))

# calculate phi
for t in range(n_topics):
    for w in range(W):
        phi[w][t] = (int(float(model_file[w][t + 1])) + beta) / (n_j[t] + W * beta)

# calculate n_.^d
n_d = sum(n_j)


total_trips = 0
# calculate theta
for t in range(n_topics):
    for d in range(D):
        total_trips += float(inf_file[d][t])
        theta[d][t] = (float(inf_file[d][t]) + alpha) / (n_d + n_topics * alpha)




