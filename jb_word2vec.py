import six
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import numpy as np
# 視覺化套件
import matplotlib
import matplotlib.pyplot as plt
# 主成分因子
from gensim.models import word2vec
from gensim import models
from sklearn.decomposition import PCA
import jieba
from matplotlib import pyplot

from nltk.cluster import KMeansClusterer
import nltk
import pandas as pd
 
from sklearn import cluster
from sklearn import metrics
# import jieba.posseg as psg


# line = '設到站提醒到底是幹嘛 可以抓好時間出門 設10分鐘提醒 剩3分鍾才提醒 又錯過公車 要去趕高鐵開會 已經很多次這樣了 可以趕快改善嗎 不然都坐計程車就好了啊'

# client = language.LanguageServiceClient()
"""
stopword_set = set()
with open('jieba_dict/stopwords.txt','r', encoding='utf-8') as stopwords:
    for stopword in stopwords:
        stopword_set.add(stopword.strip('\n'))

output = open('output_jb.txt', 'w', encoding='utf-8')

fileTrainRead = []
big = []
with open('18all.txt') as fileTrainRaw:

  for line in fileTrainRaw:
    if line != '\n':
        # print(line)
        line = line.strip('\n')

        words = jieba.cut(line,cut_all=False)
        # seg = psg.cut(line)

        # for ele in seg:
        #     print(ele)

        for word in words:
          if word not in stopword_set:
            # print(word, end=' ')
            output.write(word + ' ')
        # print()
        output.write('\n')

output.close()

sentences = word2vec.LineSentence("output_jb.txt")
model = word2vec.Word2Vec(sentences, size=300, min_count=0, negative=10)
model.save('word2vec_jb.model')
"""

model = word2vec.Word2Vec.load('word2vec_jb.model')
raw_word_vec = model.wv.vectors


# voc = model.wv.vocab
# print(list(voc))

# get vector data
X = model[model.wv.vocab]
# print (X)

clusters = list(range(26))
# print(list(range(46)))
groups = []
for i in range(26):
    groups.append([])
# =====================================Clustering====================================

NUM_CLUSTERS=26
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
print (assigned_clusters)

 

words = list(model.wv.vocab)
for i, word in enumerate(words):  
    print (word + ":" + str(assigned_clusters[i]))
    groups[assigned_clusters[i]-1].append(word)
 
dict = {"groups": clusters,  
        "words": groups
       }

select_df = pd.DataFrame(dict)

print(select_df) # 看看資料框的外觀
select_df.to_csv("jb_test.csv")
 
kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)
 
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
 
print ("Cluster id labels for inputted data")
print (labels)
print ("Centroids data")
print (centroids)
 
print ("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
print (kmeans.score(X))
 
silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
 
# 愈接近 1 表示績效愈好，反之愈接近 -1 表示績效愈差。
print ("Silhouette_score: ")
print (silhouette_score)

# 迴圈
# silhouette_avgs = []
# ks = range(2, 51)
# for k in ks:
#     kmeans_fit = cluster.KMeans(n_clusters = k, n_init = 15).fit(X)
#     cluster_labels = kmeans_fit.labels_
#     silhouette_avg = metrics.silhouette_score(X, cluster_labels)
#     silhouette_avgs.append(silhouette_avg)

# # 作圖並印出 k = 2 到 10 的績效
# plt.bar(ks, silhouette_avgs)
# plt.show()
# print(silhouette_avgs)
# ==============================================================================
# res = model.wv.most_similar(u"地點")
# for item in res: 
#       print(item[0] + ':' + str(item[1]))

"""
cent_word1 = "公車" #r
cent_word2 = "預測" #b
cent_word3 = "問題" #g
cent_word4 = "顯示" #k
cent_word5 = "地點" #c

wordList1 = model.wv.most_similar(cent_word1, topn=20)
wordList2 = model.wv.most_similar(cent_word2, topn=20)
wordList3 = model.wv.most_similar(cent_word3, topn=20)
wordList4 = model.wv.most_similar(cent_word4, topn=20)
wordList5 = model.wv.most_similar(cent_word5, topn=20)


wordList1 = np.append([item[0] for item in wordList1], cent_word1)
wordList2 = np.append([item[0] for item in wordList2], cent_word2)
wordList3 = np.append([item[0] for item in wordList3], cent_word3)
wordList4 = np.append([item[0] for item in wordList4], cent_word4)
wordList5 = np.append([item[0] for item in wordList5], cent_word5)

def get_word_index(word):
    index = model.wv.vocab[word].index
    return index

index_list1 = map(get_word_index, wordList1)
index_list2 = map(get_word_index, wordList2)
index_list3 = map(get_word_index, wordList3)
index_list4 = map(get_word_index, wordList4)
index_list5 = map(get_word_index, wordList5)

vec_reduced = PCA(n_components=2).fit_transform(raw_word_vec)
zhfont = matplotlib.font_manager.FontProperties(fname='wqy-microhei.ttc')
x = np.arange(-0.01, 0.1, 0.01)
y = x/2
plt.plot(x, y)

for i in index_list1:
    plt.text(vec_reduced[i][0], vec_reduced[i][1], model.wv.index2word[i], color='r', fontproperties=zhfont)

for i in index_list2:
    plt.text(vec_reduced[i][0], vec_reduced[i][1], model.wv.index2word[i], color='b', fontproperties=zhfont)

for i in index_list3:
    plt.text(vec_reduced[i][0], vec_reduced[i][1], model.wv.index2word[i], color='g', fontproperties=zhfont)

for i in index_list4:
    plt.text(vec_reduced[i][0], vec_reduced[i][1], model.wv.index2word[i], color='k', fontproperties=zhfont)

for i in index_list5:
    plt.text(vec_reduced[i][0], vec_reduced[i][1], model.wv.index2word[i], color='c', fontproperties=zhfont)
plt.show()
"""




