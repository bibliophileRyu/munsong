# file:///Volumes/Ryu_disk/KoNLP%20(%EC%9E%90%EC%97%B0%EC%96%B4%20%EC%B2%98%EB%A6%AC)%20(1).html

import pandas as pd
import re
from konlpy.tag import Mecab
import numpy as np
import nltk
import matplotlib as re
from openpyxl import load_workbook
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import platform
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 폰트 경로 설정
font_dirs = ['/Library/Fonts']

# 빅카인즈 2016.01.01~2021.06.31 기간동안 '문송'을 검색한 결과. (범위 : 빅카인즈에 있는 모든 언론사)

df = pd.read_excel('/Volumes/Ryu_disk/project_dj/data/문송_20160101-20210630.xlsx')


# 모든 키워드가 든 리스트 만들기
keywords = list()
for i, row in df.iterrows():
    keyword = row.키워드
    each_keyword = keyword.split(',')
    keywords.extend(each_keyword)

# 모든 기사가 든 스트링 만들기
text = str()
for i, row in df.iterrows():
    news = row.본문
    text += news

# 형태소 분석
sent_tag = list()
pos_list = list()

mecab = Mecab()
morph = mecab.pos(text)
sent_tag.append(morph)

for sent in sent_tag:
    for word, tag in sent:
        if tag in ['NNG', 'NNP']: # 품사가 명사인 것만
            pos_list.append(word)

# 불용어 처리
stop_words = ['기자', '말', '문']

tokenized_doc = [each_word for each_word in pos_list if each_word not in stop_words]


# 빈도수 상위 30개
ko = nltk.Text(tokenized_doc, name='문송')
plt.figure(figsize=(12, 6))
ko.plot(50)
plt.show()


# 빈도수 기반 워드클라우드 그리기
import stylecloud

# 기사에서 토크나이징, 품사가 명사인 것만 추린 데이터에서 보기
# data = ko.vocab().most_common(150)
# stylecloud.gen_stylecloud(text = dict(data),
#                           palette='colorbrewer.diverging.Spectral_11',
#                           background_color='black',
#                           font_path = '/Library/Fonts/AppleGothic.ttf',
#                           output_name='문송_빅카인.png',
#                           icon_name = 'fas fa-flag',
#                           gradient='horizontal')


# 빅카인즈가 제공하는 키워드 데이터에서 보기

## 키워드
ko2 = nltk.Text(keywords, name='문송2')
plt.figure(figsize=(12, 6))
ko2.plot(50)
plt.show()

data2 = ko2.vocab().most_common(150)
stylecloud.gen_stylecloud(text = dict(data2),
                          palette='colorbrewer.diverging.Spectral_11',
                          background_color='black',
                          font_path = '/Library/Fonts/AppleGothic.ttf',
                          output_name='문송_빅카인드_키워드.png',
                          icon_name = 'fas fa-flag',
                          gradient='horizontal')




#####  LDA 토픽 모델링

from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models

NUM_TOPICS = 5
NUM_TOPIC_WORDS = 30

doc_keywords = df['키워드']

def str_to_list(str):
    return str.split(',')

doc_keywords = doc_keywords.apply(str_to_list)

# 인물
df_figures = df['인물']
data3 = df_figures.dropna() # 결측치 제거
data3 = data3.apply(str_to_list)
figure_list = list()

for i in data3:
    figure_list.extend(i)


from collections import Counter


print("---Counter()---")
result = Counter(figure_list)
print(result)

for key in result:
    print(key, result[key])


## LDA 토픽 모델링
dictionary = corpora.Dictionary(doc_keywords)
corpus = [dictionary.doc2bow(text) for text in doc_keywords]

# LDA 모델 훈련시키기

import gensim
NUM_TOPICS = 20
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS,
                                           id2word = dictionary, passes = 15)

topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)

# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
# pyLDAvis.display(vis)

print(len(df))