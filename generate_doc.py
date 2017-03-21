import psycopg2
conn = psycopg2.connect("dbname=bart user=yaoyang host=localhost")
cur = conn.cursor()
cur.execute("SELECT clippercardid, randomweekid, [c, tagontime, tagofftime, tagonlocationid, tagofflocationid FROM data1 WHERE agencyid = '4';")

def time_to_index(week, day, time):
    return (int(week) - 6)*7*24 + (int(day)-1)*24 + int(time[:2])

corpus = []
count = []
for i in range(504):
    corpus.append([])
    count.append({})
for item in cur:
    word = item[5]+'_to_'+item[6]
    corpus[time_to_index(item[1],item[2],item
                         [3])].append(word)
for i, item in enumerate(corpus):
    doc = set()
    for word in item:
        if word in doc:
            count[i][word] = count[i][word] +1
        else:
            doc.add(word)
            count[i][word] = 1

f = open('corpus.txt', 'w')
for item in count:
    for key in item:
        f.write(key+' '+str(item[key])+' ')
    f.write('\n')
f.close()