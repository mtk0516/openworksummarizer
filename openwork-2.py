from flask import Flask, render_template_string,request,render_template
from flask_cors import CORS

###############dataframe html#################
html="""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>こんな雰囲気の会社ってあるのかな</title>
</head>
<body>
<body bgcolor="#afeeee" text="#000000">
<h1><b><p>{{ message }}</p></b></h1>

<div style="position:absolute; top:10px; right:10px"> 

<img src="./static/shoku.png" width="168px" height="108px" alt="いらすとや">
</div>

<form action="/" method="POST" enctype="multipart/form-data">
	<div>
		<label for="name">検索したいワード：</label>
		<input type="text" id="name" name="name" placeholder="フリーワードで入力">	
	</div>
	<div>
		<input type="submit" value="検索"><br>
		
		
	</div>

<body>
	<p>{{ 職種検索 }}</p>
	<form action="/" method="POST" enctype="multipart/form-data">
		<select name="sel">
			<option value="null" disabled selected>選択して下さい</option>
			<option value="鶏">鶏肉</option>
			<option value="豚">豚肉</option>
			<option value="牛">牛肉</option>
			<option value="羊">羊肉</option>
		</select>
		<div>
			<input type="submit" value="送信">
		</div>
	</form>
</body>


<table bgcolor="white">
<h2>検索されたワードに類似した企業文化を持つ企業(類以度が高い順)</h2>
{{table|safe}}



</body>
</html>
"""
############################################


##############FLASK config###################

app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False
#no holds cache
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

##############FLASK config###################

#########modules##################
import json
import pandas as pd
import numpy as np
import MeCab
import time
import signal
import atexit
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import csv
import numpy as np
from IPython.core.display import display
from wordcloud import WordCloud
from janome.analyzer import Analyzer
from janome.charfilter import UnicodeNormalizeCharFilter, RegexReplaceCharFilter
from janome.tokenizer import Tokenizer as JanomeTokenizer  # sumyのTokenizerと名前が被るため
from janome.tokenfilter import POSKeepFilter, ExtractAttributeFilter
import re
import ipadic

import unicodedata
#Algos
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.reduction import ReductionSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.kl import KLSummarizer
from janome.tokenizer import Tokenizer as JanomeTokenizer
from janome.analyzer import Analyzer
from janome.charfilter import *
from janome.tokenfilter import *
from sumy.utils import get_stop_words
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

import pkg_resources, imp
imp.reload(pkg_resources)
import unicodedata
import spacy
nlp = spacy.load('ja_ginza')#spacyの日本語モデル(Ginza)をロード

from ginza import *
import neologdn
import re
import emoji
import mojimoji
from collections import Counter

import sys
import pandas as pd

#
import sqlite3
####################################

##################loading and preprocessing###########################
# ------------------------------------------------------------------
sys.stderr.write("*** 開始 ***\n")
#読み込み先
file_company = "./datas/OpenWork_company.db"
file_company_info = './datas/OpenWork_company_info.db'

conn = sqlite3.connect(file_company)
conn2 = sqlite3.connect(file_company_info)
#
company_df=pd.read_sql_query('SELECT * FROM df_company', conn)
company_info_df=pd.read_sql_query('SELECT * FROM df_company_info', conn2)

#
conn.close()
conn2.close()
sys.stderr.write("*** 終了 ***\n")

#複数行に値が重複している可能性があるためdrop_duplicatesで、1行1ユニーク値にする
company_df = company_df.drop_duplicates()
company_info_df = company_info_df.drop_duplicates()

#company_dfの前処理
company_df.本文 = company_df.本文.str.replace('\u3000', ' ')
company_df.本文 = company_df.本文.str.replace('■', ' ')
company_df.本文 = company_df.本文.str.replace('⇒', ' ')
company_df.本文 = company_df.本文.str.replace('→', ' ')
company_df.本文 = company_df.本文.str.replace('□', ' ')
company_df.本文 = company_df.本文.str.replace('：', ' ')

#company_info_dfの前処理
def preprocessing(company_info_df):
        
    #まず、'--'が含まれている場合はNaNに置き換える
    replace_line = company_info_df.replace('--',None)

    #欠損していない値が3未満の企業はdrop
    processed_company_info_df = replace_line.dropna(axis=1, thresh=3,  inplace=False)

    #obj→floatに変換
    processed_company_info_df.月残業時間 = processed_company_info_df.月残業時間.astype('float')
    processed_company_info_df.有給消化率 = processed_company_info_df.有給消化率.astype('float')
    processed_company_info_df.平均年収  = processed_company_info_df.平均年収.astype('float')
    processed_company_info_df.総合スコア = processed_company_info_df.総合スコア.astype('float')
    processed_company_info_df.社員の士気 = processed_company_info_df.社員の士気.astype('float')
    processed_company_info_df.風通し = processed_company_info_df.風通し.astype('float')
    processed_company_info_df.社員相互尊重 = processed_company_info_df.社員相互尊重.astype('float')
    processed_company_info_df['20代成長性'] = processed_company_info_df['20代成長性'].astype('float')
    processed_company_info_df.長期育成 = processed_company_info_df.長期育成.astype('float')
    processed_company_info_df.コンプラ = processed_company_info_df.コンプラ.astype('float')
    processed_company_info_df.評価納得感 = processed_company_info_df.評価納得感.astype('float')
    processed_company_info_df.待遇満足度 = processed_company_info_df.待遇満足度.astype('float')
    
    #Nullは中央値で埋める
    processed_company_info_df = processed_company_info_df.fillna(value=processed_company_info_df.median())
    
    #その他、分析に必要なdfの必要な前処理
    processed_company_info_df['会社年齢'] = 2022 - processed_company_info_df['設立年']
    processed_company_info_df.drop('設立年',axis=1,inplace=True)
    
    return processed_company_info_df

company_indo_df_processed = preprocessing(company_info_df)
##################loading and preprocessing###########################



#######################mecab################################
#Mecabアナライザ
def mecab_analyzer(text):
    #正規化
    text = unicodedata.normalize("NFKC", text)
    #text = re.sub(r'\u3000','',text)
    text.replace('\u3000', '')
    tagger = MeCab.Tagger("-Owakati")
    node = tagger.parseToNode(text)

    # 指定した品詞を抽出しリストに
    word_list = []
    
    while node:
        word_type = node.feature.split(',')[0]
        if word_type in ['名詞', '形容詞', '副詞', '動詞']:
            word_list.append(node.surface)
        #ここのインデントミスると無限ループ
        node = node.next
    # リストを文字列に変換
    word_chain = ' '.join(word_list)
    return word_list
#######################################################

#stopword list 
stopword_list = []
with open('./stopwords/stopwords.csv','r',encoding='utf-8') as f:
    text = f.readlines()
    for tk in text:
        tk = tk.strip('\n')
        tk = tk.replace('\ufeff','')
        stopword_list.append(tk)


##########################main###################################

###############reading jobtypes#################
jobtype_dict = {}
with open('./jobtype.csv','r',encoding='cp932') as f:
    for k in f:
        ab = k.strip().split(',')
        if ab[0] not in jobtype_dict:
            jobtype_dict[ab[0]] = ab[1]


#職種の分類は以下の通り        
"""
その他
コーポレート・管理・総務
営業職
事務
マーケティング職
製造
研究開発
人事
開発・設計
企画系
経理財務
SE/PM系
マネジメント
エンジニア系
デザイナー
法務

"""
#以下にjobtypeの辞書を用意した
jobtype_list_dict = {0:'その他',1:"コーポレート・管理・総務",2:'営業職・販売',3:'事務',4:'マーケティング職',
                5:'製造',6:'研究開発',7:'人事',8:'開発・設計・技術',9:'生産管理・調達・物流',10:'マネジメント',11:'企画系',
                12:'現業',13:'経理財務',14:'SE/PM系',15:'エンジニア系',16:'専門職',17:'コンサルタント',18:'デザイナー',19:'法務'}


######################word search###############################


#任意の文字列の入力と、職種IDをうけとり、指定した職種IDにマッチし、入力と類似したレビューを持つ企業名とレビューを返す関数
def similar_company(input_text,jobtype=None):
    '''
    見たいレビューの職種を限定したい場合は
    以下のjobtype_listのindex番号を引数で渡し、指定すること(指定しなくてもOK)
    (基本的には「その他」は分類不能なもの(「総合職」、「本社」、「～～会社xx事業部」)が多いです)
    
jobtype_list_dict = {0:'その他',1:"コーポレート・管理・総務",2:'営業職・販売',3:'事務',4:'マーケティング職',
                5:'製造',6:'研究開発',7:'人事',8:'開発・設計・技術',9:'生産管理・調達・物流',10:'マネジメント',11:'企画系',
                12:'現業',13:'経理財務',14:'SE/PM系',15:'エンジニア系',16:'専門職',17:'コンサルタント',18:'デザイナー',19:'法務'}
    
    '''
    #tokenization
    tokens = mecab_analyzer(input_text)
    #トークンを格納したリストを作成
    token_list = []
    token_list.append(' '.join(tokens))
    sample_vector = vec.transform(token_list)

    # 計算した文書のtfidf_matrixと指定した文字列のベクトルのコサイン類似度を計算
    text_similarity = cosine_similarity(sample_vector, tfidf_matrix)
    
    # 類似度が0.25以上の要素の数を取り出す
    num_of_similarities = (np.sum([x > 0.25 for x in text_similarity]))

    #類似度でsortとargsortして、類似度とインデックス双方を取り出す
    top_indices = np.argsort(-text_similarity)[0][:num_of_similarities].tolist()
    top_similarity = np.sort(text_similarity.ravel())[::-1][:num_of_similarities].tolist()
    
    #dfのハコを作り
    answer_df = pd.DataFrame(columns=['類似度','企業名', '評価', '本文', '職種', '経験年数','現職/退職','新卒/中途','性別'])
    
    #カラム内の文字数。デフォルトは50なので変更
    pd.set_option("display.max_colwidth", 1500)
    
    #行数上限も変更し
    pd.set_option("display.max_rows", 101)

    #forループで連結する
    for a,b in zip(top_indices,top_similarity):
        index_data = company_df.iloc[a]
        answer_df =answer_df.append({'類似度': b,'企業名': index_data[0], 
                                     '評価': index_data[4],'本文':index_data[5],
                                     '職種':index_data[8],'経験年数':index_data[9],
                                     '現職/退職':index_data[10],'新卒/中途':index_data[11],
                                     '性別':index_data[12]},ignore_index=True)
        
    #出来上がったdfを、company_info_dfとマージ
    answer_df_marged = pd.merge(answer_df,pd.DataFrame(company_info_df[['企業名','総合スコア']]),on='企業名',how='inner')
    
    #企業風土の評価スコアはdrop
    #answer_df_marged = answer_df_marged.drop('評価',axis=1)
    #出来上がったdfを、類以度でsort　topは表示件数
    top = 30
    answer_df_marged_g = answer_df_marged.set_index(['総合スコア','企業名'])
    
    #jobtypeの入力に応じて出力を変える
    #会社の総合スコア順に表示
    
    if jobtype is None:
        return answer_df_marged_g.sort_index(ascending=False)[:top]
    
    else:
        answer_df_marged_g['職種'] =  answer_df_marged_g['職種'].map(jobtype_dict)
        job_adm = answer_df_marged_g[answer_df_marged_g['職種'] == jobtype_list_dict[jobtype]].reset_index()
        job_adm_s = job_adm.set_index(['総合スコア','企業名'])
        return job_adm_s.sort_index(ascending=False)[:top]


##########################main###################################

###################when process ends, delete all static datas##########
def all_done():
    #shutil.rmtreeでディレクトリごと消してから
    shutil.rmtree('./static')
    #同じディレクトリを作成することで、ファイルを擬似的に一括削除
    os.mkdir('./static')
    
###################when process ends, delete all static datas##########

##########################run###################################

@app.route('/search/',methods=["GET", "POST"])
def post():
    p = request.form.get('sel')
    name = request.form.get('name')#user input
    if name is None and p is None:
        
        output = similar_company('雰囲気が良い')
        
        return (render_template_string(html,table=output.to_html(header='true'),message='雰囲気が良い'))
        
    else:
        output = similar_company(name)
        
        return (render_template_string(html,table=output.to_html(header='true'),message=name))

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response

if __name__ == '__main__':
    app.run(host='127.0.0.1',debug=True,port='5000')
#############################################################