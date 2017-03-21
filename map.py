'''
 Colma                       | 1        |37.684638, -122.466233
 Embarcadero (BART)          | 10       |37.792905, -122.397059
 West Oakland                | 11       |37.804687, -122.29513
 Oakland City Center 12th St | 12       |37.803769, -122.271451
 19th St Oakland             | 13       |37.808072, -122.268845
 MacArthur                   | 14       |37.829064, -122.267047
 Rockridge                   | 15       |37.844702, -122.251392
 Orinda                      | 16       |37.844702, -122.251392
 Lafayette                   | 17       |37.893734, -122.124298
 Walnut Creek                | 18       |37.905480, -122.067153
 Pleasant Hill               | 19       |37.928675, -122.055708
 Daly City                   | 2        |37.706092, -122.469032
 Concord (BART)              | 20       |37.974359, -122.029186
 North Concord/Martinez      | 21       |38.003193, -122.024656
 Pittsburg/Bay Point         | 22       |38.018916, -121.945133
 Ashby                       | 23       |37.852913, -122.269542
 Downtown Berkeley           | 24       |37.870097, -122.268147
 North Berkeley              | 25       |37.873977, -122.283402
 El Cerrito Plaza            | 26       |37.902643, -122.298946
 El Cerrito Del Norte        | 27       |37.924894, -122.317052
 Richmond                    | 28       |37.936309, -122.35382
 Lake Merritt                | 29       |37.797030, -122.265183
 Balboa Park                 | 3        |37.721612, -122.447519
 Fruitvale                   | 30       |37.774835, -122.224182
 Coliseum/Oakland Airport    | 31       |37.753669, -122.196898
 San Leandro                 | 32       |37.721950, -122.160855
 Bay Fair                    | 33       |37.696975, -122.126581
 Hayward                     | 34       |37.669790, -122.087009
 South Hayward               | 35       |37.634642, -122.0566
 Union City                  | 36       |37.590626, -122.017393
 Fremont                     | 37       |37.557468, -121.976634
 Castro Valley               | 38       |37.690761, -122.075533
 Dublin/Pleasanton           | 39       |37.701650, -121.899181
 Glen Park                   | 4        |37.733067, -122.433829
 South San Francisco (BART)  | 40       |37.663798, -122.443571
 San Bruno (BART)            | 41       |37.630490, -122.411084
 San Francisco Airport       | 42       |37.615963, -122.392415
 Millbrae (BART)             | 43       |37.600271, -122.386707
 West Dublin                 | 44       |37.700747, -121.927256
 24th St Mission             | 5        |37.752476, -122.418146
 16th St Mission             | 6        |37.764812, -122.420011
 Civic Center (BART)         | 7        |37.779756, -122.41415
 Powell St (BART)            | 8        |37.784469, -122.407986
 Montgomery (BART)           | 9        |37.789348, -122.401144
'''
import numpy as np
import networkx as nx


with open('1.txt','rt') as f:
    con = f.read().split()
dic = {}
for i in range(44):
    dic[int(con[i*4+1])]=[float(con[i*4+2]),float(con[i*4+3])]

direction = {}
with open('direction.txt','rt') as f:
    for line in f:
        dir = line.split()
        direction[dir[0]] = 1

def getweight(v1, v2):
    return np.sqrt((dic[v1][0]-dic[v2][0])**2+(dic[v1][1]-dic[v2][1])**2)

def get_map():
    bart_map = nx.Graph()
    for i in range(44):
        bart_map.add_node(i+1)

    edges = [(43,41,getweight(43,41)),(42,41,getweight(42,41)),(41,40,getweight(41,40)),(40,1,getweight(40,1))]
    for i in range(1,22):
        edges.append((i,i+1,getweight(i,i+1)))
    edges.append((14,23,getweight(14,23)))
    for i in range(23,28):
        edges.append((i,i+1,getweight(i,i+1)))
    edges.append((11,29,getweight(11,29)))
    edges.append((12,29,getweight(12,29)))
    for i in range(29,37):
        edges.append((i, i + 1, getweight(i, i + 1)))
    edges.append((33,38,getweight(33,38)))

    edges.append((38, 44, getweight(38, 44)))
    edges.append((44, 39, getweight(44, 39)))


    bart_map.add_weighted_edges_from(edges)
    # print(bart_map.edges(data='weight'))
    #  print(nx.shortest_path(bart_map,10,22,weight='weight'))
    return bart_map

def get_unweighted_map():
    bart_map = nx.Graph()
    for i in range(44):
        bart_map.add_node(i+1)

    edges = [(43,41,0),(42,41,0),(41,40,0),(40,1,0)]
    for i in range(1,22):
        edges.append((i,i+1,0))
    edges.append((14,23,0))
    for i in range(23,28):
        edges.append((i,i+1,0))
    edges.append((11,29,0))
    edges.append((12,29,0))
    for i in range(29,37):
        edges.append((i, i + 1, 0))
    edges.append((33,38,0))

    edges.append((38, 44, 0))
    edges.append((44, 39, 0))


    bart_map.add_weighted_edges_from(edges)
    # print(bart_map.edges(data='weight'))
    #  print(nx.shortest_path(bart_map,10,22,weight='weight'))
    return bart_map


def get_poly_line(G):
    edges = G.edges(data='weight')
    poly_lines =[]
    for item in edges:
        poly_lines.append([[tuple(dic[item[0]]),tuple(dic[item[1]])],item[2]])

    return poly_lines



def add_count(G_1, G_2 , M, word ,counts):
    line = word.split("_")
    a = int(line[0])
    b = int(line[2])
    edges=(nx.shortest_path(M,a,b,weight='weight'))
    if len(edges) == 1:
        return [G_1,G_2]
    for i in range(len(edges)-1):
        if get_direction(edges[i],edges[i+1]) == 1:
            G_1.edge[edges[i]][edges[i+1]]['weight'] += counts
        else:
            G_2.edge[edges[i]][edges[i + 1]]['weight'] += counts
    return [G_1,G_2]

def get_direction(a,b):
    # 1 for map 1
    # 2 for map 2

    word_1 = str(a)+'_'+str(b)
    if word_1 in direction:
        return 1
    else:
        return 2


# print(dic[3])
# a = get_unweighted_map()
# poly = get_poly_line(a)
# for item in poly:
#     print(item)
# print(a.edges())