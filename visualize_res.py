import numpy as np
import matplotlib.pyplot as plt
import map
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import networkx as nx
import folium
import glob
import PIL.Image
from selenium import webdriver
import os

res = []
count = []
dic = {}
inf = []
bart_map = map.get_map()
count_map_1 = []
count_map_2 = []

#
# def get_intensity(a):
#
#     if a < 100:
#         return '#ffb2b2'
#     elif a <300:
#         return '#ff7f7f'
#     elif a <500:
#         return '#ff4c4c'
#     elif a < 800:
#         return '#ff3232'
#     else:
#         return '#ff0000'

def get_intensity(a):

    if a < 100:
        return '#ffb2b2'
    elif a <300:
        return '#ff7f7f'
    elif a <500:
        return '#ff4c4c'
    elif a < 800:
        return '#ff3232'
    else:
        return '#ff0000'




for i in range(504):
    count_map_1.append([])
    count_map_2.append([])
for i in range(504):
    for j in range(4):
        count_map_1[i].append(map.get_unweighted_map())
        count_map_2[i].append(map.get_unweighted_map())
with open('/Users/Martin/Desktop/plda/tmp/model_01001.txt','r') as f:
    for line in f:
        res.append(line)

with open('/Users/Martin/Desktop/plda/tmp/inference_01001.txt','r') as f:
    for line in f:
        inf.append(line)

n_topic = 4


with open('corpus.txt','rt') as f:
    for line in f:
        count.append(line)
# dic -- prob of word given different topics
for i in range(len(res)):
    word = res[i].split()
    sum = 0
    for j in range(n_topic):
        sum += int(word[j+1])
    dic[word[0]] = []
    if sum != 0:
        for j in range(n_topic):
            dic[word[0]].append(int(word[1+j])/sum)
    else:
        for j in range(n_topic):
            dic[word[0]].append(0)

# inference -- prob of different topics in each doc
inference = np.zeros((len(inf),n_topic))
for i in range(len(inf)):
    line = inf[i].split()
    sum = 0
    for j in range(n_topic):
        inference[i,j]=line[j]
        sum += float(line[j])
    for j in range(n_topic):
        inference[i,j] = inference[i,j]/sum
#
# for i in range(n_topic):
#     plt.plot(inference[:,i])
# plt.show()

final = np.zeros((504,n_topic))
n = 0
for i in range(len(count)):
    line = count[i].split()
    n_ele = len(line)/2
    if n_ele == 0:
        continue
    for j in range(int(n_ele)):

        for k in range(n_topic):
            # word_counts = dic[line[2 * j]][k] * int(line[int(2 * j + 1)]) * inference[n, k]
            word_counts =int(line[int(2 * j + 1)]) * inference[n, k]
            # [count_map_1[i][k], count_map_2[i][k]] = map.add_count(count_map_1[i][k], count_map_2[i][k], bart_map, line[2 * j],
            #                                                  word_counts)
            final[i, k] += word_counts
    n += 1
    sums = 0

    for item in count_map_1[i][0].edges(data='weight'):
        sums += item[2]

    sums = 0
    for item in count_map_2[i][0].edges(data='weight'):
        sums += item[2]

pattern = []
# nx.draw(count_map[1])
# plt.show()

colors = ['#ff1c14','#287199','#fefe85','#9accbb']

#
# poly_lines = map.get_poly_line(count_map_1[0][0])
# m = folium.Map(location=[37.774929, -122.419416], zoom_start=10,tiles='Stamen Toner')
# for item in poly_lines:
#     folium.PolyLine(locations=item[0], weight=5, color='red', opacity=1).add_to(m)
# folium.Marker(location=[37.792905, -122.397059],popup='Embarcadero').add_to(m)
# m.save('map.html')


# for i in range(24,48):
#     for j in range(4):
#         poly_lines = map.get_poly_line(count_map_1[i][j])
#         m = folium.Map(location=[37.774929, -122.419416], zoom_start=10,tiles='Stamen Toner')
#         for item in poly_lines:
#             folium.PolyLine(locations=item[0], weight=3+item[1]/80,color=colors[j], opacity=1).add_to(m)
#         folium.CircleMarker(location=[37.792905, -122.397059],radius=(i%4+1)*50,color=colors[j],fill_color=colors[j],fill_opacity=0.1).add_to(m)
#         m.save('html_res_'+str(j+1)+'/out'+str(i)+'.html')
#     for j in range(4):
#         poly_lines = map.get_poly_line(count_map_2[i][j])
#         m = folium.Map(location=[37.774929, -122.419416], zoom_start=10, tiles='Stamen Toner')
#         for item in poly_lines:
#             folium.PolyLine(locations=item[0], weight=3 + item[1] / 100, color=colors[j], opacity=1).add_to(m)
#         folium.CircleMarker(location=[37.792905, -122.397059], radius=(4-i % 4+1) * 30, color=colors[j],
#                             fill_color=colors[j], fill_opacity=0.1).add_to(m)
#         m.save('html_res_' + str(j + 1) + '/in' + str(i) + '.html')


#
for i in range(n_topic):
    pattern.append(final[:168,i].T.tolist())
    # pattern.append(final[24:48,i].T.tolist())
plt.hold(True)
for i in range(n_topic):
    plt.plot(pattern[i],label=str(i+1))
plt.legend()
# plt.locator_params(nbins=8)

ax = plt.gca()
# ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.xaxis.set_major_locator(plt.FixedLocator([12,36,60,84,108,132,156]))
ax.set_xticklabels(['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'])

plt.show()

# print(final)
# for word in res[0].split():
#     print(word)
# for word in count[0].split():
#     print(word)