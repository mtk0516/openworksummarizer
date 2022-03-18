from flask import Flask, render_template_string,request,render_template,url_for,redirect
from flask_cors import CORS

###############dataframe html#################
html="""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>あの会社ってどうなのよ</title>
</head>
<body>
<body bgcolor="#BAD3FF" text="#000000">
<h1><b><p>{{ message }}</p></b></h1>
<h2><a href="{{ url_for('func2') }}">フリーワードで会社の雰囲気を検索する機能はこちら</a></h2><br>
<div style="position:absolute; top:10px; right:10px"> 

<img src="./static/shoku.png" width="168px" height="108px" alt="いらすとや">
</div>
</body>

<form action="/" method="POST" enctype="multipart/form-data">
	<div>
		<label for="name">企業名：</label>
		<input type="text" id="name" name="name" placeholder="企業名を指定された名称で入力">	
	</div>
	<div>
		<input type="submit" value="検索"><br>
		<li><a href="./static/compname.xlsx">収録企業名一覧(xlsxファイル)</a></li>
		このExcelに記載された企業名で検索しないとエラーになります<br>
		※検索ボタンを押してから反映まで10秒~30秒程かかります。連打するとサービスがダウンしますので気長にお待ち下さい
	</div>
</form>
<tr style ="background-color: white">
<h2>指定企業の分野別評価平均</h2>


<head>
    <meta charset="utf-8">
    <title>mystyle</title>
    <link rel="stylesheet" 
        type="text/css"
        href="{{url_for('static', filename='style.css')}}">
    <style type="text/css">
        body {
                font-family: Arial, Verdana, sans-serif;
                font-size: 90%;
                color: #2b2b2b;
                background-color: #ffffff;}
        </style>
</head>
<body>



{{table|safe}}

<h2>指定企業の属する業界の分野別評価平均</h2>
{{table2|safe}}

<h2>指定企業の企業文化の要約</h2>
<p>{{culture}}</p>

<h2>指定企業の企業文化の可視化(WordCloud)</h2>
<img src={{img}} alt="WordCloud" width="420" height="315">

<h2>企業文化のレビューが類似した企業(類似度が高い順)</h2>
{{table3|safe}}
</tr>
</body>
</html>
"""
############################################


###############similartext html#################
html2="""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>こんな雰囲気の会社あるのかな？</title>
</head>
<body>
<body bgcolor="#BAD3FF" text="#000000">
<h3><b>検索ワード：<p>{{ message }}</p></b></h3>

<div style="position:absolute; top:10px; right:10px"> 

<img src="./static/shoku.png" width="168px" height="108px" alt="いらすとや">
</div>

<form action="/search" method="POST" enctype="multipart/form-data">
	<div>
		<label for="name">検索したいワード：</label>
		<input type="text" id="atmos" name="atmos" placeholder="フリーワードで入力"><br>
		※「雰囲気が良い」「若手が活躍している」・・・　など。<br>単語ではなく、長い文章の方がマッチします
		<br>無関係だったり、真反対の意味のレビューを拾うこともありますが仕様ですorz<br>(例:「残業が少ない」で検索すると、「残業が多い」もひっかかる)
	</div>
	<div>
		<input type="submit" value="検索"><br>
		
	</div>
<tr style ="background-color: white">
<h2>検索されたワードに類似した企業文化を持つ企業</h2>
※個々のコメントの評価ではなく企業としての総合平均評価(総合スコア)が高い順に25件表示

<head>
    <meta charset="utf-8">
    <title>mystyle</title>
    <link rel="stylesheet" 
        type="text/css"
        href="{{url_for('static', filename='style.css')}}">
    <style type="text/css">
        body {
                font-family: Arial, Verdana, sans-serif;
                font-size: 90%;
                color: #2b2b2b;
                background-color: #ffffff;}
        </style>
</head>

{{table4|safe}}
</tr>
<br>
<a href="{{ url_for('func1') }}">戻る</a>

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
import pickle
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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

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

##############stopwords################
#stopword list 
stopword_list = []
with open('./stopwords/stopwords.csv','r',encoding='utf-8') as f:
    text = f.readlines()
    for tk in text:
        tk = tk.strip('\n')
        tk = tk.replace('\ufeff','')
        stopword_list.append(tk)
##############stopwords################


###########pickle object fix################
class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'Manager':
            from settings import Manager
            return Manager
        return super().find_class(module, name)
###########pickle object fix################




##########################main###################################

#algos
algorithm_dic = {"lex": LexRankSummarizer(), "tex": TextRankSummarizer(), "lsa": LsaSummarizer(),\
                 "kl": KLSummarizer(), "luhn": LuhnSummarizer(), "redu": ReductionSummarizer(),\
                 "sum": SumBasicSummarizer()}

#日本語処理用のクラス
class JapaneseCorpus:
    # ①
    def __init__(self):
        self.nlp = spacy.load('ja_ginza')
        self.analyzer = Analyzer(
            char_filters=[UnicodeNormalizeCharFilter(), RegexReplaceCharFilter(r'[(\)「」、。]', ' ')],  # ()「」、。は全てスペースに置き換える
            tokenizer=JanomeTokenizer(),
            token_filters=[POSKeepFilter(['名詞', '形容詞', '副詞', '動詞']), ExtractAttributeFilter('base_form')]  # 名詞・形容詞・副詞・動詞の原型のみ
        )

    # ②
    def preprocessing(self, text):
        text = re.sub(r'\n', '', text)
        text = re.sub(r'\r', '', text)
        #text = re.sub(r'\d', '', text)
        text = re.sub(r'\s', '', text)
        text.replace('\u3000', '')
        text = text.lower()
        text = mojimoji.zen_to_han(text, kana=True)
        text = mojimoji.han_to_zen(text, digit=False, ascii=False)
        text = ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)
        text = neologdn.normalize(text)

        return text

    # ③
    def make_sentence_list(self, sentences):
        doc = self.nlp(sentences)
        self.ginza_sents_object = doc.sents
        sentence_list = [s for s in doc.sents]

        return sentence_list

    # ④
    def make_corpus(self):
        corpus = [' '.join(self.analyzer.analyze(str(s))) + '。' for s in self.ginza_sents_object]

        return corpus

#WordCloudで可視化と、レビューの要約双方を行うクラス
class company_summarize:

    def __init__(self,company_name):
        self.company_name = company_name
        self.company_id = company_df[company_df.企業名 == company_name].企業ID.unique()[0]#compane_nameをIDに変換
        self.company_df = pd.DataFrame(company_df[company_df.企業名 == self.company_name])
        self.nlp = spacy.load('ja_ginza')
        self.analyzer = Analyzer(
            char_filters=[UnicodeNormalizeCharFilter(), RegexReplaceCharFilter(r'[(\)「」、。]', ' ')],  # ()「」、。は全てスペースに置き換える
            tokenizer=JanomeTokenizer(),
            token_filters=[POSKeepFilter(['名詞', '形容詞', '副詞', '動詞']), ExtractAttributeFilter('base_form')]  # 名詞・形容詞・副詞・動詞の原型のみ
        )

    
    #Wordcloudでの可視化
    def visualize(self):
        
        tagger = MeCab.Tagger('-Owakati')
        #null parse
        tagger.parse('')
        
        #make str from marked company df
        all_text = self.company_df.本文.str.cat()
    
        #正規化
        all_text_norm = unicodedata.normalize("NFKC", all_text)
    
        #parse
        node = tagger.parseToNode(all_text_norm)

        # 指定した品詞を抽出しリストに
        word_list = []
    
        while node:
            word_type = node.feature.split(',')[0]
            if word_type in ["名詞", "動詞", "形容詞",'副詞']:
                word_list.append(node.surface)
            #ここのインデントミスると無限ループ
            node = node.next

        # リストを文字列に変換
        word_chain = ' '.join(word_list)
    
        #ストップワードリストをコピー
        stopword_list_indef = stopword_list
        
        #word_listの要素数の10%は、最も登場頻度が高いワードなのでストップリストに追加する
        fdist = Counter(word_list)
        Common_Words = fdist.most_common(n=int(len(fdist)*0.10))
        
        for common_word in Common_Words:
            stopword_list_indef.append(common_word[0])

        #社名が入らないように、stopwordlistに追加
        stopword_list_indef.append(self.company_df.企業名.drop_duplicates().str.cat().replace('株式会社',''))

        # ワードクラウド作成
        W = WordCloud(width=600, height=450, background_color='black', colormap='cool_r',
                      font_path='./font/msgothic.ttc', stopwords = set(stopword_list_indef)).generate(word_chain)
        # 表示設定
        plt.figure(figsize = (15, 12))
        plt.axis('off')
        plt.imshow(W)
        plt.savefig('./static/cloud_pic.jpg', bbox_inches='tight', pad_inches=0.0)
        
        
    #前処理、コーパスリスト作成など
    def preprocessing(self, text):
        text = re.sub(r'\n', '', text)
        text = re.sub(r'\r', '', text)
        #text = re.sub(r'\d', '', text)
        text = re.sub(r'\s', '', text)
        text.replace('\u3000', '')
        text = text.lower()
        text = mojimoji.zen_to_han(text, kana=True)
        text = mojimoji.han_to_zen(text, digit=False, ascii=False)
        text = ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)
        text = neologdn.normalize(text)

        return text

    def make_sentence_list(self, sentences):
        doc = self.nlp(sentences)
        self.ginza_sents_object = doc.sents
        sentence_list = [s for s in doc.sents]

        return sentence_list

    def make_corpus(self):
        corpus = [' '.join(self.analyzer.analyze(str(s))) + '。' for s in self.ginza_sents_object]

        return corpus
    
    #文章内容要約
    def summarize_sentences(self,sentences_count=10, algorithm="lex", language="japanese"):
        
        #レビュー件数があまりに多いとinputerrorになるので、件数を制限する
        if self.company_df.shape[0] >= 45:
            self.company_df = self.company_df.sample(n=40)
        
        all_text = self.company_df.本文.str.cat()
        all_text_norm = unicodedata.normalize("NFKC", all_text)
        sentences=''.join(all_text_norm)
        
        #JPコーパスクラス継承
        corpus_maker = JapaneseCorpus()
        preprocessed_sentences = corpus_maker.preprocessing(sentences)
        preprocessed_sentence_list = corpus_maker.make_sentence_list(preprocessed_sentences)
        corpus = corpus_maker.make_corpus()
        parser = PlaintextParser.from_string(" ".join(corpus), Tokenizer(language))

        try:
            summarizer = algorithm_dic[algorithm]
        except KeyError:
            print("algorithm name:'{}'is not found.".format(algorithm))

        summarizer.stop_words = get_stop_words(language)
        #sentences_countは文書に一定の割合をかけた値でもよいが、読みやすさ重視で10センテンス固定長とする
        summary = summarizer(document=parser.document, sentences_count=10)
        
        if language == "japanese":
            return str("".join([str(preprocessed_sentence_list[corpus.index(sentence.__str__())]) for sentence in summary]))
        else:
            return str("".join([sentence.__str__() for sentence in summary]))

######################company similarity########################
# モデルのロード(モデルが用意してあれば、ここからで良い)
m = Doc2Vec.load("./model/openwork_uniquecompany.model")

#企業ごとコメント全部をベクトル化する
#企業毎の本文連結dfをpklから読み込む
all_docs = pd.read_pickle("./static/allcompany_df.pkl")

#企業名を引数にとり、その企業のコメントベクトルに類似したベクトルを持つ企業名と類似度を表示する関数
def company_comparison(company_name):
    identified_company_docs = all_docs[all_docs.index == company_name]
    input_text = identified_company_docs.本文.str.cat()
    #Mecabによって形態素解析
    mecab_input = mecab_analyzer(input_text)
    
    #docvecsによる類似性判定結果
    #類似度TOP10の企業を表示
    similarity = (m.docvecs.most_similar(positive=[m.infer_vector(mecab_input)],topn=10))
    
    #格納用空リスト
    return_list=[]
    for num,similarity in similarity:
        return_list.append([all_docs.index[num],similarity])
    #df化
    output_df = pd.DataFrame(return_list,columns=['企業名','類似度']).sort_values('類似度',ascending=False)
    return output_df[1:] #自社が最も類似しているので、1行目は除いて、2行目から表示
    
######################company similarity########################
    
###############reading jobtypes#################
jobtype_dict = {}
with open('./static/jobtype.csv','r',encoding='cp932') as f:
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


#############all text token list###########



######################word search###############################
#任意の文字列の入力と、職種IDをうけとり、指定した職種IDにマッチし、入力と類似したレビューを持つ企業名とレビューを返す関数
def similar_company(input_text,jobtype=None):
    
    '''
    見たいレビューの職種を限定したい場合は
    以下のjobtype_listのindex番号を引数で渡し、指定すること(指定しなくてもOK)
    (基本的には「その他」は分類不能なもの(「総合職」、「本社」、「～～会社xx事業部」)が多いです)
    '''
    jobtype_list_dict = {0:'その他',1:"コーポレート・管理・総務",
                         2:'営業職・販売',3:'事務',4:'マーケティング職',
                         5:'製造',6:'研究開発',7:'人事',8:'開発・設計・技術',
                         9:'生産管理・調達・物流',10:'マネジメント',11:'企画系',
                         12:'現業',13:'経理財務',14:'SE/PM系',15:'エンジニア系',
                         16:'専門職',17:'コンサルタント',18:'デザイナー',19:'法務'}
                         
                         
    #tfidf_matrixをpickleから読み込み
    tfidf_matrix = pickle.load(open("./static/tfidf_matrix.pkl", "rb"))
    #tfidf_matrix = CustomUnpickler(open('./static/tfidf_matrix.pkl', 'rb')).load()
    #tfidf_matrix.set_params(analyzer=mecab_analyzer)
    
    #TfIdfインスタンス作成
    vec = TfidfVectorizer(analyzer=mecab_analyzer,
                          stop_words=stopword_list,
                          min_df=10,max_df=0.05,#登場頻度上位5%の単語は対象外
                          ngram_range=(1,3))#bigram/unigram/Triramを考慮する
    
    #pickleから作成したvecを読み込む
    vec = CustomUnpickler(open('./static/tfidf_vec.pkl', 'rb')).load()
    vec.set_params(analyzer=mecab_analyzer)
    
    #tokenization
    tokens = mecab_analyzer(input_text)
    #トークンを格納したリストを作成
    token_list_input = []
    token_list_input.append(' '.join(tokens))
    
    #トークンを格納したリストを作成
    #token_list_input = []
    #token_list_input.append(' '.join(tokens))
    sample_vector = vec.transform(token_list_input)

    # 計算した文書のtfidf_matrixと指定した文字列のベクトルのコサイン類似度を計算
    text_similarity = cosine_similarity(sample_vector, tfidf_matrix)
    
    # 類似度が0.3以上の要素の数を取り出す
    num_of_similarities = (np.sum([x > 0.3 for x in text_similarity]))

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
    top = 25
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

@app.route('/',methods=["GET","POST"],endpoint='func1')
def func1():
    
    name = request.form.get('name')#user input
    if name is None:
        C = company_summarize(company_name = 'グーグル合同会社')
        #企業文化の要約
        summary = C.summarize_sentences()
        #類似企業の表示
        similar_companies = company_comparison('グーグル合同会社')
        #Wordcloud表示
        crowd_visual = C.visualize()
        
        #初期企業(google)の平均スコア
        company_eval = company_indo_df_processed[company_indo_df_processed.企業名 == C.company_name].drop('業種',axis=1)

        #該当企業の業種平均
        company_info_mean = company_indo_df_processed.groupby('業種').agg(np.mean).round(2).reset_index()
        company_industry_avg = company_info_mean[company_info_mean.業種 == company_info_df[company_info_df.企業名 == C.company_name].業種.str.cat()]

        return (render_template_string(html,table=company_eval.to_html(header='true'),table2=company_industry_avg.to_html(header='true'),table3=similar_companies,message=name,culture=summary,img='./static/cloud_pic.jpg'))
        
    elif request.method == 'POST' and name is not None:
        C = company_summarize(name)
        #企業文化の要約
        summary = C.summarize_sentences()
        #類似企業の表示
        similar_companies = company_comparison(name)
        #Wordcloudの表示
        crowd_visual = C.visualize()
        
        #該当企業の平均スコア
        company_eval = company_indo_df_processed[company_indo_df_processed.企業名 == C.company_name].drop('業種',axis=1)

        #該当企業の業種平均
        company_info_mean = company_indo_df_processed.groupby('業種').agg(np.mean).round(2).reset_index()
        company_industry_avg = company_info_mean[company_info_mean.業種 == company_info_df[company_info_df.企業名 == C.company_name].業種.str.cat()]

        return (render_template_string(html,table=company_eval.to_html(header='true'),table2=company_industry_avg.to_html(header='true'),table3=similar_companies,message=name,culture=summary,img='./static/cloud_pic.jpg'))
        
@app.route('/search',methods=["GET","POST"],endpoint='func2')
def func2():
    user_input = request.form.get('atmos')#user input

    if request.method == 'POST' and user_input is not None:
        output = similar_company(input_text=user_input)
        return (render_template_string(html2,table4=output.to_html(header='true'),message=user_input))
    
    else:
        return (render_template_string(html2))
    
            

if __name__ == '__main__':
    app.run(host='127.0.0.1',debug=True,port='5000')
#############################################################