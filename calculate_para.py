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
n_topics = 3

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
if n_topics == 5:
    topic_order[0]=np.argmax(np.array(inf_file[31], dtype=float))
    topic_order[1]=np.argmax(np.array(inf_file[41], dtype=float))
    topic_order[2]=np.argmax(np.array(inf_file[12], dtype=float))
    topic_order[3]=np.argmax(np.array(inf_file[17], dtype=float))
    topic_order[4] = np.argmax(np.array(inf_file[32], dtype=float))
if n_topics == 4:
    topic_order[0]=np.argmax(np.array(inf_file[31], dtype=float))
    topic_order[1]=np.argmax(np.array(inf_file[41], dtype=float))
    topic_order[2]=np.argmax(np.array(inf_file[12], dtype=float))
    topic_order[3]=np.argmax(np.array(inf_file[17], dtype=float))
if n_topics == 3:
    topic_order[0] = np.argmax(np.array(inf_file[31], dtype=float))
    topic_order[1] = np.argmax(np.array(inf_file[41], dtype=float))
    topic_order[2] = np.argmax(np.array(inf_file[17], dtype=float))

# print(np.array(inf_file, dtype=float).sum(axis=0)/np.array(inf_file, dtype=float).sum())
#
# morning = np.array(inf_file, dtype=float)[:, topic_order[0]].reshape((21,24)).sum(axis = 1)
# evening = np.array(inf_file, dtype=float)[:, topic_order[1]].reshape((21,24)).sum(axis = 1)
# print('morning topic: ')
# print(morning)
# print('evening topic: ')
# print(evening)



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


def get_para():
    return theta, phi

def get_order():
    return topic_order

def get_model_dic():
    return model_dic



# p(z=j|w,d)
# res = np.zeros((D, W, n_topics))
# for d in range(D):
#     for w in range(W):
#         for t in range(n_topics):
#             res[d][w][t] = theta[d][t] * phi[w][t]
# for d in range(D):
#     for w in range(W):
#         sums = sum(res[d][w])
#         res[d][w] /= sums
# for d in range(D):
#     print(res[d][model_dic['3_to_10']])


# res = res * total_trips


#


# f = plt.figure()
#
# for i in range(9):
#     data = res[33+i, :, 1]
#     rs = np.zeros((44, 44))
#     print(np.sum(data))
#     for j in range(44):
#         for k in range(44):
#             word = od_to_word(j + 1, k + 1)
#             if word in model_dic:
#                 rs[j, k] = data[model_dic[word]]
#             else:
#                 rs[j, k] = 0
#     ax = f.add_subplot(331 + i)
#     ax.set_xticks(np.arange(0, 44, 1))
#     ax.set_yticks(np.arange(0, 44, 1))
#     ax.set_xticklabels(np.arange(1, 45, 1))
#     ax.set_yticklabels(np.arange(1, 45, 1))
#     ax.set_xticks(np.arange(-.5, 44, 1), minor=True)
#     ax.set_yticks(np.arange(-.5, 44, 1), minor=True)
#     ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
#     ax.set_title('hour:'+str(i+11))
#     cax = ax.imshow(rs)
# # plt.colorbar(cax)
# plt.show()



# phi = P(w|z)


# od_matrix = {}
# rs1 = np.zeros((44, 44))
# for i in range(n_topics):
#     order = topic_order[i]
#     f = plt.figure(figsize=(15,15))
#     data = np.array(phi[:, order])
#     data1 = overall
#     rs = np.zeros((44, 44))
#     for j in range(44):
#         for k in range(44):
#             word = od_to_word(j + 1, k + 1)
#             if word in model_dic:
#                 rs[j, k] = data[model_dic[word]]
#                 rs1[j, k] = data1[model_dic[word]]
#             else:
#                 rs[j, k] = 0
#                 rs1[j, k] = 0
#     od_matrix[order] = rs
#     ax = f.add_subplot(111)
#     ax.set_xticks(np.arange(0, 44, 1))
#     ax.set_yticks(np.arange(0, 44, 1))
#     ax.set_xticklabels(np.arange(1, 45, 1))
#     ax.set_yticklabels(np.arange(1, 45, 1))
#     ax.set_xticks(np.arange(-.5, 44, 1), minor=True)
#     ax.set_yticks(np.arange(-.5, 44, 1), minor=True)
#     ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
#     ax.set_title('topic:'+str(i+1))
#
#     if n_topics == 4:
#         if i == 0:
#             cax = ax.imshow(rs,vmin=0, vmax=0.02)
#         elif i == 1:
#             cax = ax.imshow(rs, vmin=0, vmax=0.018)
#         elif i == 2:
#             cax = ax.imshow(rs, vmin=0, vmax=0.035)
#         else:
#             cax = ax.imshow(rs, vmin=0, vmax=0.05)
#     else:
#         cax = ax.imshow(rs)
#     plt.colorbar(cax)
#     plt.savefig('image2/'+str(i+1)+'topic_'+num+'.png')
# f = plt.figure(figsize=(15, 15))
# ax = f.add_subplot(111)
# ax.set_xticks(np.arange(0, 44, 1))
# ax.set_yticks(np.arange(0, 44, 1))
# ax.set_xticklabels(np.arange(1, 45, 1))
# ax.set_yticklabels(np.arange(1, 45, 1))
# ax.set_xticks(np.arange(-.5, 44, 1), minor=True)
# ax.set_yticks(np.arange(-.5, 44, 1), minor=True)
# ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
# ax.set_title('overall:' + str(i + 1))
# cax = ax.imshow(rs1)
# plt.colorbar(cax)
# plt.savefig('image2/' + str(i + 1) + 'overall_' + num + '.png')
# if n_topics == 3:
#
#     print('KL divergence: ')
#     print('  ','      1       ','      2       ','      3       ')
#     print('1 ','**************',kl_div(od_matrix[topic_order[0]],od_matrix[topic_order[1]]),kl_div(od_matrix[topic_order[0]],od_matrix[topic_order[2]]))
#     print('2 ','**************','**************',kl_div(od_matrix[topic_order[1]],od_matrix[topic_order[2]]))
#     print('3 ','**************','**************','**************')
#     print('KL divergence between topic 1 and transpose (topic 2)')
#
#
# if n_topics == 4:
#     print('KL divergence: ')
#     print('  ','      1       ','      2       ','      3       ','      4       ')
#     print('1 ','**************',kl_div(od_matrix[topic_order[0]],od_matrix[topic_order[1]]),kl_div(od_matrix[topic_order[0]],od_matrix[topic_order[2]]),kl_div(od_matrix[topic_order[0]],od_matrix[topic_order[3]]))
#     print('2 ','**************','**************',kl_div(od_matrix[topic_order[1]],od_matrix[topic_order[2]]),kl_div(od_matrix[topic_order[1]],od_matrix[topic_order[3]]))
#     print('3 ','**************','**************','**************',kl_div(od_matrix[topic_order[2]],od_matrix[topic_order[3]]))
#     print('4 ','**************','**************','**************','**************')
#     print('KL divergence between topic 1 and transpose (topic 2)')
# print(kl_div(od_matrix[topic_order[0]],np.array(np.matrix(od_matrix[topic_order[1]]).T)))
#
#
#
# if n_topics == 5:
#     print('KL divergence: ')
#     print('  ','      1       ','      2       ','      3       ','      4       ','      5       ')
#     print('1 ','**************',kl_div(od_matrix[topic_order[0]],od_matrix[topic_order[1]]),kl_div(od_matrix[topic_order[0]],od_matrix[topic_order[2]]),kl_div(od_matrix[topic_order[0]],od_matrix[topic_order[3]]),kl_div(od_matrix[topic_order[0]],od_matrix[topic_order[4]]))
#     print('2 ','**************','**************',kl_div(od_matrix[topic_order[1]],od_matrix[topic_order[2]]),kl_div(od_matrix[topic_order[1]],od_matrix[topic_order[3]]),kl_div(od_matrix[topic_order[1]],od_matrix[topic_order[4]]))
#     print('3 ','**************','**************','**************',kl_div(od_matrix[topic_order[2]],od_matrix[topic_order[3]]),kl_div(od_matrix[topic_order[2]],od_matrix[topic_order[4]]))
#     print('4 ','**************','**************','**************','**************',kl_div(od_matrix[topic_order[3]],od_matrix[topic_order[4]]))
#     print('5 ','**************','**************','**************','**************','**************')
#     print('KL divergence between topic 1 and transpose (topic 2)')
#     print(kl_div(od_matrix[topic_order[2]],np.array(np.matrix(od_matrix[topic_order[3]]).T)))

# print(LA.norm(np.matrix(od_matrix[0]) - np.matrix(od_matrix[1]).T))



'''

4 topics/ a = 0.1 / b = 0.01
KL divergence:
         1              2              3              4
1  ************** 0.534900162427 0.270372724287 0.49544232171
2  ************** ************** 0.453570286942 0.273494601032
3  ************** ************** ************** 0.379466214282
4  ************** ************** ************** **************
KL divergence between topic 1 and transpose (topic 2)
0.0269794350679


4 topics/ a = 1 / b = 0.01
KL divergence:
         1              2              3              4
1  ************** 0.54276931552 0.259258629039 0.489926829008
2  ************** ************** 0.47719807293 0.275816752544
3  ************** ************** ************** 0.367104241503
4  ************** ************** ************** **************
KL divergence between topic 1 and transpose (topic 2)
0.0403335053844

4 topics/ a = 10 / b = 0.01
[ 0.25647848 0.31201752 0.22380161 0.20770239]
morning topic:
[   3099.6
    35549.6
    36993.
    36143.2
    36482.6
    32009.4
    4882.8
    3263.
    35655.4
    35506.4
    37121.2
    35992.2
    31261.4
    4902.
    3810.8
    35421.8
    37127.8
    36441.4
    36593.6
    31757.8
    4199.6]
evening topic:
[   2595.6
    43949.6
    45801.2
    44811.8
    45050.4
    38751.6
    4520.4
    2828.
    43637.6
    43526.
    45647.2
    44416.8
    38159.2
    4895.6
    3364.6
    43602.
    46012.2
    44840.
    45129.4
    38671.6
    4016. ]
KL divergence:
         1              2              3              4
1  ************** 0.558162818597 0.285414610269 0.519593372649
2  ************** ************** 0.483477345291 0.286033817055
3  ************** ************** ************** 0.388796400422
4  ************** ************** ************** **************
KL divergence between topic 1 and transpose (topic 2)
0.0406307381953

3 topics/ a = 10 / b = 0.01
[ 0.38449036  0.3472505  0.26825915]
morning topic:
[   6376.
    51777.4
    54059.2
    52821.
    53828.6
    48266.6
     10160.4
     6472.8
  51559.4
  50828.8
  54583.4
  53658.4
  47141.8
  10581.8
  7109.8
  51466.6
  54509.
  53963.2
  54144.2
  48317.
  9205.2]
evening topic:
[  4181.
48049.2
50115.4
48970.6
49544.
43254.4
6954.8
4403.8
  47804.4
  47117.2
  50525.
  49040.
  42005.6
  7411.8
  5140.4
  47290.4
  50480.4
  49368.6
  49603.4
  42868.4
  6231.6]
KL divergence:
         1              2              3
1  ************** 0.547899283431 0.335136135657
2  ************** ************** 0.326281600748
3  ************** ************** **************
KL divergence between topic 1 and transpose (topic 2)
0.0181840692115

3 topics/ a = 1 / b = 0.01
KL divergence:
         1              2              3
1  ************** 0.526950995311 0.314828441877
2  ************** ************** 0.298791508512
3  ************** ************** **************
0.0156879377347

3 topics/ a = 0.1 / b = 0.01
KL divergence:
         1              2              3
1  ************** 0.521893834387 0.313209677232
2  ************** ************** 0.279835919498
3  ************** ************** **************
0.0161651224418

3 topics/ a = 100 / b = 0.01
KL divergence:
         1              2              3
1  ************** 0.586372120709 0.391143217234
2  ************** ************** 0.390347576642
3  ************** ************** **************
0.0187545271537

3 topics/ a = 10 / b = 0.1
KL divergence:
         1              2              3
1  ************** 0.481502339907 0.301087948598
2  ************** ************** 0.242146328017
3  ************** ************** **************
0.0129634142433

3 topics/ a = 10 / b = 1
KL divergence:
         1              2              3
1  ************** 0.484694678573 0.300909211161
2  ************** ************** 0.246350990258
3  ************** ************** **************
0.0120296243594

5 topics/ a = 10 / b = 0.01
KL divergence:
         1              2              3              4              5
1  ************** 0.518140423983 0.31474048652 0.511200435454 0.186198170084
2  ************** ************** 0.371827263854 0.203078664721 0.476598387096
3  ************** ************** ************** 0.307752337705 0.217900996417
4  ************** ************** ************** ************** 0.425905781955
5  ************** ************** ************** ************** **************
KL divergence between topic 1 and transpose (topic 2)
0.0647298843362
KL divergence between topic 3 and transpose (topic 4)
0.0897670147302
'''

'''
 weekid | weekday | count
--------+---------+-------
 6      | 1       |  7319
 6      | 2       | 38500
 6      | 3       | 40175
 6      | 4       | 39233
 6      | 5       | 38281
 6      | 6       | 32514
 6      | 7       |  8866
 7      | 1       |  6608
 7      | 2       | 38166
 7      | 3       | 38008
 7      | 4       | 39067
 7      | 5       | 38507
 7      | 6       | 32759
 7      | 7       |  9816
 8      | 1       |  6573
 8      | 2       | 38641
 8      | 3       | 39423
 8      | 4       | 39764
 8      | 5       | 38485
 8      | 6       | 32433
 8      | 7       |  9242
'''