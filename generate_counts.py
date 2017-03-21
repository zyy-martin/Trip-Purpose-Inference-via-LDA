from calculate_para import get_order, get_para, get_model_dic
import numpy as np
import re
import csv


theta, phi = get_para()
order = get_order()
dic = get_model_dic()
corpus_dir= '/Users/Martin/Desktop/Fall2016/plda/testdata/corpus_013.txt'



H_W_on = np.zeros((44,504))
W_H_on = np.zeros((44,504))
other_on =  np.zeros((44,504))
H_W_off = np.zeros((44,504))
W_H_off = np.zeros((44,504))
other_off =  np.zeros((44,504))


corpus_file = []
with open(corpus_dir, 'rt') as f:
    for line in f:
        line = re.split(('\n|\t| '), line)
        line.remove('')
        line.pop()
        corpus_file.append(line)


for i in range(504):
    for j in range(int(len(corpus_file[i])/2)):
        word = corpus_file[i][j*2]
        word = word.split('_')
        origin = word[0]
        dest = word[2]
        H_W_theta = theta[i,order[0]]
        W_H_theta = theta[i,order[1]]
        Other_theta = theta[i,order[2]]
        H_W_phi = phi[:,order[0]]
        W_H_phi = phi[:, order[1]]
        Other_phi = phi[:, order[2]]
        z =  H_W_theta * H_W_phi[dic[origin+'_to_'+dest]] +  W_H_theta * W_H_phi[dic[origin+'_to_'+dest]] + Other_theta * Other_phi[dic[origin+'_to_'+dest]]
        H_W_on[int(origin)-1,i] += int(int(corpus_file[i][j*2+1]) * H_W_theta * H_W_phi[dic[origin+'_to_'+dest]]/z)
        W_H_on[int(origin)-1,i] += int(int(corpus_file[i][j*2+1]) * W_H_theta * W_H_phi[dic[origin+'_to_'+dest]]/z)
        other_on[int(origin)-1,i] += int(int(corpus_file[i][j*2+1]) * Other_theta * Other_phi[dic[origin+'_to_'+dest]]/z)

        H_W_off[int(dest)-1,i] += int(int(corpus_file[i][j*2+1]) * H_W_theta * H_W_phi[dic[origin+'_to_'+dest]]/z)
        W_H_off[int(dest)-1,i] += int(int(corpus_file[i][j*2+1]) * W_H_theta * W_H_phi[dic[origin+'_to_'+dest]]/z)
        other_off[int(dest)-1,i] += int(int(corpus_file[i][j*2+1]) * Other_theta * Other_phi[dic[origin+'_to_'+dest]]/z)

stop_dict = {}
with open('stops.txt','rt') as f:
    lines = f.readlines()[1:]
    for line in lines:
        line = line.split(',')
        stop_dict[int(line[11])]=[line[0],line[1],line[3],line[4]]

with open('tagon_H_W.csv', 'wt') as file:
    writer = csv.writer(file, delimiter=',')
    title = ['week', 'day', 'hour']
    for i in range(44):
        title.append(stop_dict[i+1][1])
    writer.writerow(title)
    for i in range(504):
        row = []
        row.append(int(i/168)+1)
        row.append(int((i%168)/24)+1)
        row.append((i%168)%24+1)
        for j in range(44):
            element = H_W_on[j, i]
            row.append(element)
        writer.writerow(row)

with open('tagon_W_H.csv', 'wt') as file:
    writer = csv.writer(file, delimiter=',')
    title = ['week', 'day', 'hour']
    for i in range(44):
        title.append(stop_dict[i+1][1])
    writer.writerow(title)
    for i in range(504):
        row = []
        row.append(int(i/168)+1)
        row.append(int((i%168)/24)+1)
        row.append((i%168)%24+1)
        for j in range(44):
            element = W_H_on[j, i]
            row.append(element)
        writer.writerow(row)

with open('tagon_other.csv', 'wt') as file:
    writer = csv.writer(file, delimiter=',')
    title = ['week', 'day', 'hour']
    for i in range(44):
        title.append(stop_dict[i+1][1])
    writer.writerow(title)
    for i in range(504):
        row = []
        row.append(int(i/168)+1)
        row.append(int((i%168)/24)+1)
        row.append((i%168)%24+1)
        for j in range(44):
            element = other_on[j, i]
            row.append(element)
        writer.writerow(row)

with open('tagoff_H_W.csv', 'wt') as file:
    writer = csv.writer(file, delimiter=',')
    title = ['week', 'day', 'hour']
    for i in range(44):
        title.append(stop_dict[i + 1][1])
    writer.writerow(title)
    for i in range(504):
        row = []
        row.append(int(i / 168) + 1)
        row.append(int((i % 168) / 24) + 1)
        row.append((i % 168) % 24 + 1)
        for j in range(44):
            element = H_W_off[j, i]
            row.append(element)
        writer.writerow(row)

with open('tagoff_W_H.csv', 'wt') as file:
    writer = csv.writer(file, delimiter=',')
    title = ['week', 'day', 'hour']
    for i in range(44):
        title.append(stop_dict[i + 1][1])
    writer.writerow(title)
    for i in range(504):
        row = []
        row.append(int(i / 168) + 1)
        row.append(int((i % 168) / 24) + 1)
        row.append((i % 168) % 24 + 1)
        for j in range(44):
            element = W_H_off[j, i]
            row.append(element)
        writer.writerow(row)

with open('tagoff_other.csv', 'wt') as file:
    writer = csv.writer(file, delimiter=',')
    title = ['week', 'day', 'hour']
    for i in range(44):
        title.append(stop_dict[i + 1][1])
    writer.writerow(title)
    for i in range(504):
        row = []
        row.append(int(i / 168) + 1)
        row.append(int((i % 168) / 24) + 1)
        row.append((i % 168) % 24 + 1)
        for j in range(44):
            element = other_off[j, i]
            row.append(element)
        writer.writerow(row)