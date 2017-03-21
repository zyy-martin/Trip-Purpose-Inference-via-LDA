import numpy as np
import os
import re
import matplotlib.pyplot as plt
import random

# 001 -- 0.5 alpha 0.1 beta 0.01 random
# 002 -- 0.5 alpha 1 beta 0.01 random
# 024 -- 0 alpha 1 beta 0.01 random
# 003 -- 0.5 alpha 10 beta 0.01 random
# 004 -- 0 alpha 10 beta 0.01 random
# 005 -- 0.5 alpha 20 beta 0.01 random
# 006 -- 0.5 alpha 10 beta 0.01 random
# 007 -- 0.3 alpha 10 beta 0.01 random
# 008 -- 0.1 alpha 10 beta 0.01 random
# 009 -- 0.7 alpha 10 beta 0.01 random
# 010 -- 0.9 alpha 10 beta 0.01 random
# 011 -- 0.5 alpha 10 beta 0.01 random 400
# 012 -- 0.5 alpha 10 beta 0.01 random 800


# 013 -- 0   alpha 10 beta 0.01 topic 3
# 014 -- 0.5   alpha 10 beta 0.01 topic 3
# 015 -- 0.7   alpha 10 beta 0.01 topic 3
# 016 -- 0.9   alpha 10 beta 0.01 topic 3

# 017 -- 0   alpha 1 beta 0.01 topic 3
# 018 -- 0   alpha 0.1 beta 0.01 topic 3
# 019 -- 0   alpha 0.01 beta 0.01 topic 3
# 020 -- 0   alpha 100 beta 0.01 topic 3

# 021 -- 0   alpha 10 beta 0.1 topic 3
# 022 -- 0   alpha 10 beta 1 topic 3
# 023 -- 0   alpha 10 beta 10 topic 3

# 025 -- 0   alpha 10 beta 0.01 topic 5
# 026 -- 0   alpha 10 beta 0.01 topic 6
# 026 -- 0   alpha 10 beta 0.01 topic 7


number = '013'
week_index = {}



parent_dir = '/Users/Martin/Desktop/Fall2016/plda/testdata/'
corpus_dir= '/Users/Martin/Desktop/Fall2016/plda/testdata/corpus_'+number+'.txt'
inf_dir = '/Users/Martin/Desktop/Fall2016/plda/tmp/inf_'+number+'.txt'




t_model_dir ='tmp/model_'+number+'.txt'
t_inf_dir = 'tmp/inf_'+number+'.txt'
t_corpus_dir = 'testdata/corpus_'+number+'.txt'


def generate_index():
    for i in range(120):
        week_index[i] = i + 24
        week_index[i + 120] = i + 120 + 48
        week_index[i + 240] = i + 240 + 72

def get_corpus(corpus_dir= '/Users/Martin/Desktop/Fall2016/plda/testdata/corpus.txt'):
    corpus = []
    with open(corpus_dir, 'rt') as f:
        for line in f:
            line = re.split(('\n|\t| '), line)
            line.remove('')
            line.pop()
            dic = {}
            for i in range(int(len(line) / 2)):
                dic[line[2 * i]] = int(line[2 * i + 1])
            corpus.append(dic)
    return corpus


def get_inference(inf_dir, n_topics, corpus_dir='/Users/Martin/Desktop/Fall2016/plda/testdata/corpus.txt'):
    corpus = get_corpus(corpus_dir)
    inf_file_temp = []
    inf_file = []
    with open(inf_dir, 'rt') as f:
        for line in f:
            line = line.split()
            inf_file_temp.append(line)
    count_c = 0
    count_i = 0
    size = len(corpus)
    while count_c < size:
        if len(corpus[count_c]) != 0 :
            inf_file.append(inf_file_temp[count_i])
            count_c += 1
            count_i += 1
        else:
            inf_file.append([0] * n_topics)
            count_c += 1
    return inf_file


def merge(dic1, dic2):
    for key in dic2:
        if key in dic1:
            dic1[key] += dic2[key]
        else:
            dic1[key] = dic2[key]
    return dic1


def shuffle(corpus, hour1, hour2, ratio=0.5):
    dic1 = corpus[hour1]
    dic2 = corpus[hour2]
    dic3 = {}
    dic4 = {}
    for key in dic1:
        dic3[key] = int(dic1[key] * ratio)
        dic1[key] = dic1[key]-dic3[key]
    for key in dic2:
        dic4[key] = int(dic2[key] * ratio)
        dic2[key] = dic2[key]-dic4[key]
    return merge(dic1, dic4), merge(dic2, dic3)


def save_file(dir, name, corpus):
    target = dir + name
    with open(target,'wt') as f:
        for doc in corpus:
            for key in doc:
                f.write(key+' '+str(doc[key])+' ')
            f.write('\n')


def plot_result(inf_dir, n_topics, name, corpus_dir= '/Users/Martin/Desktop/Fall2016/plda/testdata/corpus.txt'):
    inf = np.array(get_inference(inf_dir, n_topics,corpus_dir))

    for i in range(n_topics):
        plt.plot(inf[:168, i], label=str(i+1))
    plt.legend()
    plt.ylim(0,16000)
    os.chdir('/Users/Martin/Desktop/Fall2016/bart')
    plt.savefig('img/'+name+'.png')
    plt.show()


def generate_model(corpus_dir, model_dir):
    print('start generating model...')
    os.chdir('/Users/Martin/Desktop/Fall2016/plda')
    os.system(
        './lda --num_topics 3 --alpha 10 --beta 0.01 --training_data_file '+corpus_dir+' --model_file '+model_dir+' --burn_in_iterations 100 --total_iterations 150')


def generate_inf(corpus_dir, model_dir, inf_dir):
    print('start inferring...')
    os.chdir('/Users/Martin/Desktop/Fall2016/plda')
    os.system('./infer --alpha 10 --beta 0.01 --inference_data_file '+corpus_dir+' --inference_result_file '+inf_dir+' --model_file '+model_dir+' --total_iterations 15 --burn_in_iterations 10')


# generate_index()
# corpus = get_corpus()

# swap with 1 week after

# for i in range(24, 144):
#     temp1, temp2 = shuffle(corpus,i,i+168,0.5)
#     corpus[i] = temp1
#     corpus[i+168] = temp2
# save_file(parent_dir, 'corpus_'+number+'.txt',corpus)
# generate_model(t_corpus_dir,t_model_dir)
# generate_inf(t_corpus_dir,t_model_dir,t_inf_dir)
#

# randomly swap
# for i in range(0):
#     num1 = random.randint(0, 359)
#     int2 = random.randint(1,2)
#     num2 = (week_index[num1]+int2 * 168) % 504
#     temp1, temp2 = shuffle(corpus, week_index[num1], num2, 0.5)
#     corpus[week_index[num1]] = temp1
#     corpus[num2] = temp2
# save_file(parent_dir, 'corpus_'+number+'.txt',corpus)
# generate_model(t_corpus_dir,t_model_dir)
# generate_inf(t_corpus_dir,t_model_dir,t_inf_dir)


# plot

# plot_result(inf_dir, 3, number, corpus_dir)
# plot_result('/Users/Martin/Desktop/Fall2016/plda/tmp/inf_013.txt',3,'013',corpus_dir)
# plot_result('/Users/Martin/Desktop/Fall2016/plda/tmp/inference_01001.txt',4,number+'_original','/Users/Martin/Desktop/Fall2016/plda/testdata/corpus.txt')



