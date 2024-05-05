#!/usr/bin/env python
# coding: utf-8

# In[47]:


#基本ライブラリ
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from datetime import timedelta
import random
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import os

pd.set_option('display.max_rows', 100) #最大100行まで表示
pd.set_option('display.max_columns', 100)#最大100列まで表示
#pd.options.display.float_format = '{:.5f}'.format #floatの表記を統一
pd.options.display.precision = 15


# In[1]:


class dataset():
    
    
    def __init__(self,kigyo_cd,weeksday):
        
        self.kigyo_cd = kigyo_cd
        self.weeksday = weeksday
        self.ym = self.weeksday[:4] + self.weeksday[5:7]
       
        #特徴量マートのカラムの情報
        feature_cols_path = '//Np2/resource/各種プロジェクト/店頭KIOSK端末/feature_特徴量テーブル/data_archive/columns.csv'
        self.feature_cols = pd.read_csv(feature_cols_path,header=None,encoding='shift_jis')
        self.feature_cols.columns = ["row","cname","cname_jap","col_no"]
    
    #================================================================================================
    
    #0.
    #--------------------------------------------------#
    # 関数名:read_csv
    # 引数:input_flg(blue:モデル入力用→True 学習用→False)
    # 戻り値:なし
    # 処理内容:SQLで集計したcsvをデータフレームdfで読み込む
    #--------------------------------------------------#    
    
    def read_csv(self,input_flg = True):
    
        #入力用なのか
        if input_flg:
            data_path = f'csv/{self.kigyo_cd}_mltest_input.csv'
        else:
            data_path = f'csv/eo_{self.kigyo_cd}_data.csv'
            
        dtype = {0:'str',
                 1:'str',
                 2:'str', 
                 3:'str',
                 4:'int',
                 5:'str',
                 6:'str',
                 7:'int',
                 8:'float',
                 9:'float',
                 10:'str',
                 11:'float',
                 12:'str',
                 13:'float',
                 14:'float',
                 15:'float',
                 16:'float',
                 17:'float',
                 18:'float'}
        cols = ['cust_cd','weeksday','ym','set_date','days_remaining','ab','coupon','ptweek_flg',
                'coupon_1week_ago','coupon_2week_ago','ymd_last_kaiage','days_last_kaiage','ymd_last_use',
                'days_last_use','urikngk_cumsum','use_cnt_cumsum','overshoot_delta','urikngk_month','get_point']
        
        self.df = pd.read_csv(data_path,header=None,dtype = dtype,usecols = range(19))
        self.df.columns = cols
      
     #================================================================================================
    
    #1.
    #--------------------------------------------------#
    # 関数名:drop_data
    # 引数:なし
    # 戻り値:なし
    # 処理内容:dfで不必要なレコードを消去する
    #--------------------------------------------------#    
    
    def drop_data(self):

        #A会員のみに限定してよい(仮定1)
        self.df = self.df[self.df['ab']=='A']
        
        #(入力週以外)買上があった人に限定してよい(仮定1)
        self.df = self.df[(self.df['urikngk_month'].notnull())|(self.df["weeksday"] == self.weeksday)]
        
        #ドラモリ様は2021-10-13の週で表示されず，本来の分布を汚すので思い切って捨てる
        if self.kigyo_cd == '061':
            self.df = self.df[self.df["weeksday"] != '2021-10-13'].reset_index(drop= True)
            
        #coupon = 9999は、少しだけ使用してもう使用していないので思い切って捨てる
        self.df = self.df[self.df["coupon"] != '9999'].reset_index(drop= True)
        
        #coupon_1week_agoやcoupon_2week_agoは欠損扱いにする
        self.df["coupon_1week_ago"] = self.df["coupon_1week_ago"].where(self.df["coupon_1week_ago"] != 9999.0, -1)
        self.df["coupon_2week_ago"] = self.df["coupon_2week_ago"].where(self.df["coupon_2week_ago"] != 9999.0, -1)
            
    #================================================================================================
    
    #2.
    #--------------------------------------------------#
    # 関数名:pre_merge
    # 引数:x(data_reader),cust(dataframe[cust_cd,ym])
    #     ,usecols(list:特徴量マートのカラム番号)
    # 戻り値:x(custとjoin後)
    # 処理内容:Memory Errorを回避するための前処理関数
    #--------------------------------------------------#    
    
    def pre_merge(self,x,cust,usecols):
        x.columns = self.feature_cols[self.feature_cols["row"].isin(usecols)]["cname"].tolist() #カラム名はfeature_colsから取得
        x = pd.merge(cust,x,how = 'inner',on = ["cust_cd"])
        return x
        
    #================================================================================================
    
    #3.
    #--------------------------------------------------#
    # 関数名:read_feature
    # 引数:usecols(list:特徴量マートのカラム番号)
    #      input_flg(blue:モデル入力用→True 学習用→False)
    # 戻り値:なし
    # 処理内容:特徴量マートテーブルを読み込み，dfとjoin
    #--------------------------------------------------#    
    
    def read_feature(self,usecols,input_flg = True):
        
        #結合する特徴量マートの月を取得(ドラモリ様は7日、ザグザグ様は6日までは先月分の特徴量マートを参照できない)
        
        self.df['ym_num'] = pd.to_datetime(self.df["weeksday"]).dt.year * 12 + pd.to_datetime(self.df["weeksday"]).dt.month - 2
        self.df['year_num'] = self.df['ym_num']//12
        self.df['month_num'] = self.df['ym_num']%12 + 1
        self.df['ym_feature'] = self.df['year_num'].astype('str') + ('0' + self.df['month_num'].astype('str')).str[-2:]

        self.df['ym_num_1month_ago'] = self.df['ym_num'] - 1
        self.df['year_num_1month_ago'] = self.df['ym_num_1month_ago']//12
        self.df['month_num_1month_ago'] = self.df['ym_num_1month_ago']%12 + 1
        self.df['ym_feature_1month_ago'] = self.df['year_num_1month_ago'].astype('str') + ('0' + self.df['month_num_1month_ago'].astype('str')).str[-2:]

        self.df['daily'] = pd.to_datetime(self.df["weeksday"]).dt.day

        #金曜日が3日以降なら特徴量マートの情報を使える(2日に集計できるので)
        if self.kigyo_cd == '061':
            day_feature = 8
        elif self.kigyo_cd == '109':
            day_feature = 7

        self.df.loc[self.df['daily'] < day_feature, 'ym_feature'] = self.df['ym_feature_1month_ago']
        self.df = self.df.drop(['ym_num','year_num','month_num','ym_num_1month_ago','year_num_1month_ago',
                      'month_num_1month_ago','ym_feature_1month_ago','daily'],axis = 1)
        
        #対象週の月を取得
        kijun_day = datetime.date(int(self.ym[:4]), int(self.ym[4:]), 1)
        #入力用データの取得開始は2021/9月(高額クーポン運用開始)
        #学習用データの取得開始は2021/1月
        
        if input_flg:
            data_start_day = kijun_day
        else:
            data_start_day = datetime.date(2021, 9, 1)
        
        #取得開始月からの経過月数
        keika_month = (kijun_day.year - data_start_day.year)*12 + kijun_day.month - data_start_day.month + 1
        
        #参照する特徴量マート月リストを作成
        #list_monthly = [str(kijun_day + relativedelta(months=-n-1)).replace("-", "")[:6] for n in range(keika_month+1)]
        list_monthly = self.df['ym_feature'].drop_duplicates().tolist()
        
        #空のデータフレーム作成
        cols = ["cust_cd","ym_feature"] + self.feature_cols[self.feature_cols["row"].isin(usecols)]["cname"].tolist()[1:]
        feature = pd.DataFrame(columns = cols)

        #月×custごとに特徴量をつける※例えば12月のdfには11月の特徴量マートデータをぶつけたいので注意
        for ym in list_monthly:

            #特徴量マートのパス(ex.11月～7月)
            path = f'//Np2/resource/各種プロジェクト/店頭KIOSK端末/feature_特徴量テーブル/data_archive/{self.kigyo_cd}/np_cust_feature_{self.kigyo_cd}_{ym}.csv'
            #データがなければ(月マタギで特徴量マートのデータがなければ)仕方なく昨月分を使う
            if not os.path.exists(path):
                path = f'//Np2/resource/各種プロジェクト/店頭KIOSK端末/feature_特徴量テーブル/data_archive/{self.kigyo_cd}/np_cust_feature_{self.kigyo_cd}_{list_monthly[n+2]}.csv'
            
            #custリスト(dataframe型だけど)を作成
            cust = self.df[self.df['ym_feature']==ym][['cust_cd','ym_feature']].drop_duplicates()
            #10,000行ずつ前処理関数をかけながら読み込む
            data_reader = pd.read_csv(path, chunksize=10000,header = None,usecols = usecols,dtype = {2:'str'}, encoding='shift_jis')
            #結合
            feature_monthly = pd.concat((self.pre_merge(x,cust,usecols) for x in data_reader), ignore_index=True)
            del data_reader
            #月ごとのdataframeを追加
            feature = feature.append(feature_monthly,ignore_index = True)
            del feature_monthly
    
        #dfに特徴量マートデータを追加
        self.df = pd.merge(self.df,feature,on = ["cust_cd","ym_feature"],how = 'left')
        
        self.df = self.df.drop(['ym_feature'],axis = 1)
    
    #================================================================================================
    
    #4.
    #--------------------------------------------------#
    # 関数名:preprocess
    # 引数:input_flg(blue:モデル入力用→True 学習用→False)
    # 戻り値:なし
    # 処理内容:dfの前処理を実行する
    #--------------------------------------------------#    
    
    def preprocess(self,usecols,input_flg = True):
    
        #(1)getpointは0埋め
        self.df["get_point"] = self.df["get_point"].fillna(0)

        #(2)1週前や2週前の設定データがnullなら-1を入れる(あとでnullを0埋めするのでそれと区別)
        self.df['coupon_1week_ago'] = self.df['coupon_1week_ago'].fillna(-1)
        self.df['coupon_2week_ago'] = self.df['coupon_2week_ago'].fillna(-1)

        #(3)weeksdayから「何週目か」の情報を取得
        self.df['weeks_no'] = pd.to_datetime(self.df["weeksday"]).dt.day // 7
        #old::day_remainingは日にちの情報だが、データ数がすくない内は(2か月分)過学習を恐れて週数の情報に落とそう
        #self.df["days_remaining"] = self.df["days_remaining"]//7
        #self.df["days_remaining"] = self.df["days_remaining"]>=16
        
        #days_remainingの情報は落としておく
        self.df = self.df.drop(['days_remaining'],axis = 1)

        #(4)クーポン発券実績や利用実績は3か月分の和をとっちゃう
        self.df["cpnout_eo_sum"] = self.df["cpnout_eo_sum_1st"].fillna(0)+self.df["cpnout_eo_sum_2nd"].fillna(0)+self.df["cpnout_eo_sum_3rd"].fillna(0)
        self.df["cpnuse_eo_sum"] = self.df["cpnuse_eo_sum_1st"].fillna(0)+self.df["cpnuse_eo_sum_2nd"].fillna(0)+self.df["cpnuse_eo_sum_3rd"].fillna(0)
        self.df["cpnuse_shouhin_atari_sum"] = self.df["cpnuse_shouhin_atari_sum_1st"].fillna(0)+self.df["cpnuse_shouhin_atari_sum_2nd"].fillna(0)+self.df["cpnuse_shouhin_atari_sum_3rd"].fillna(0)
        self.df["cpnuse_tab_sum"] = self.df["cpnuse_tab_sum_1st"].fillna(0)+self.df["cpnuse_tab_sum_2nd"].fillna(0)+self.df["cpnuse_tab_sum_3rd"].fillna(0)
        
        self.df = self.df.drop(["cpnout_eo_sum_1st","cpnout_eo_sum_2nd","cpnout_eo_sum_3rd",
                                "cpnuse_eo_sum_1st","cpnuse_eo_sum_2nd","cpnuse_eo_sum_3rd",
                                "cpnuse_shouhin_atari_sum_1st","cpnuse_shouhin_atari_sum_2nd","cpnuse_shouhin_atari_sum_3rd",
                                "cpnuse_tab_sum_1st","cpnuse_tab_sum_2nd","cpnuse_tab_sum_3rd"
                                 ],
                               axis = 1)
        
        if set([102,103,104]) <= set(usecols): #集合の包含関係
            self.df["tab_touch_week_cnt"] = self.df["tab_touch_week_cnt_1st"].fillna(0)+self.df["tab_touch_week_cnt_2nd"].fillna(0)+self.df["tab_touch_week_cnt_3rd"].fillna(0)
            self.df = self.df.drop(["tab_touch_week_cnt_1st",
                                    "tab_touch_week_cnt_2nd",
                                    "tab_touch_week_cnt_3rd"
                                     ],
                                   axis = 1)

        #(5)ログイン日数は3か月分、購買日数と買い上げ金額は後ろ2か月分をまとめる
        self.df["buy_day_cnt_23"] = self.df["buy_day_cnt_2nd"].fillna(0)+self.df["buy_day_cnt_3rd"].fillna(0)
        self.df["buy_urikngk_sum_23"] = self.df["buy_urikngk_sum_2nd"].fillna(0)+self.df["buy_urikngk_sum_3rd"].fillna(0)
        self.df["login_day_cnt"] = self.df["login_day_cnt_1st"].fillna(0)+self.df["login_day_cnt_2nd"].fillna(0)+self.df["login_day_cnt_3rd"].fillna(0)

        self.df = self.df.drop(["buy_day_cnt_2nd","buy_day_cnt_3rd",
                                "buy_urikngk_sum_2nd","buy_urikngk_sum_3rd",
                                "login_day_cnt_1st","login_day_cnt_2nd","login_day_cnt_3rd"
                                ],
                               axis = 1)

        #(6)クーポン設定フラグ
        if not input_flg:
            self.df['set_flg'] = self.df['coupon'] != '0'
            self.df['set_flg'] = self.df['set_flg'].astype(int)

        #(7)高額クーポンフラグ
        if not input_flg:
            self.df['high_flg'] = self.df['coupon'].astype(int) >= 6000
            self.df['high_flg'] = self.df['high_flg'].astype(int)
            
        #(8)last_dayは92で埋める(のちに0埋めするので混同しないようにするため)
        self.df['days_last_kaiage'] = self.df['days_last_kaiage'].fillna(92)
        self.df['days_last_use'] = self.df['days_last_use'].fillna(92)
        
        #(9)test_no(機械学習クラスタとランダムクラスタがいるので、分布がことなることを考慮)
        
        if self.kigyo_cd == '061':
        
            self.df['mod7'] = self.df['cust_cd'].astype(int) % 7         
            self.df['test_no'] = 0 #通常クラスタ
            self.df.loc[(self.df['mod7'] == 2)&(self.df['weeksday']>='2021-12-22'), 'test_no'] = 1 #機械学習クラスタ(反実仮想)
            self.df.loc[(self.df['mod7'] == 5)&(self.df['weeksday']>='2022-02-09'), 'test_no'] = 1 #機械学習クラスタ(bandit)
            self.df.loc[(self.df['mod7'] == 3)&(self.df['weeksday']>='2022-04-06'), 'test_no'] = 1 #機械学習クラスタ(反実仮想)
            self.df.loc[(self.df['mod7'] == 6)&(self.df['weeksday']>='2022-04-06'), 'test_no'] = 1 #機械学習クラスタ(bandit)
            self.df.loc[(self.df['mod7'] == 0)&(self.df['weeksday']>='2022-06-01'), 'test_no'] = 1 #機械学習クラスタ(bandit)
            self.df.loc[(self.df['mod7'] == 1)&(self.df['weeksday']>='2022-01-26'), 'test_no'] = 1 #randomクラスタも1にしてpscoreの下限を高くするべし
            self.df = self.df.drop(['mod7'],axis = 1)
            
        elif self.kigyo_cd == '109':
            
            self.df['mod13'] = self.df['cust_cd'].astype(float) % 13
            self.df['mod13'] = self.df['mod13'].astype(int)        
            self.df['test_no'] = 0 #通常クラスタ
            #2022/01月の運用
            self.df.loc[(self.df['mod13'].isin([0,1,2,3,4,5,6,9,11,12]))&(self.df['weeksday']>='2021-12-28')&(self.df['weeksday']<='2022-01-25'), 'test_no'] = 1 #機械学習クラスタ
            self.df.loc[(self.df['mod13'].isin([7,8,10]))&(self.df['weeksday']>='2021-12-28')&(self.df['weeksday']<='2022-01-25'), 'test_no'] = 1 #randomクラスタも1にしないと
            #2022/02月以降の運用
            self.df.loc[(self.df['mod13'].isin([0,3,6,11,12]))&(self.df['weeksday']>='2022-02-01')&(self.df['weeksday']<='2022-04-26'), 'test_no'] = 1 #機械学習クラスタ
            self.df.loc[(self.df['mod13'].isin([8]))&(self.df['weeksday']>='2022-02-01'), 'test_no'] = 1 #randomクラスタ
            #2022/05月以降の運用
            self.df.loc[(self.df['mod13'].isin([0,3,5,6,7,10,11,12]))&(self.df['weeksday']>='2022-05-03')&(self.df['weeksday']<='2022-05-31'), 'test_no'] = 1 #機械学習クラスタ
            self.df.loc[(self.df['mod13'].isin([8]))&(self.df['weeksday']>='2022-02-01'), 'test_no'] = 1 #randomクラスタ
            #2022/06月以降の運用
            self.df.loc[(self.df['mod13'].isin([0,1,3,5,6,7,10,11,12]))&(self.df['weeksday']>='2022-06-07'), 'test_no'] = 1 #機械学習クラスタ
            self.df = self.df.drop(['mod13'],axis = 1)
            
        '''
        #(8)学習用データは、allと9月以降で分ける
        if not input_flg:
            self.df_all = self.df.copy()
            self.df = self.df[self.df['ym']>='202109'].reset_index(drop = True)
        '''
    
    #================================================================================================
    
    #0.
    #--------------------------------------------------#
    # 関数名:check
    # 引数:DF(dataframe)
    # 戻り値:DFの情報をもつdataframe
    # 処理内容:dfの情報を確認するための関数
    #--------------------------------------------------#    
    
    def check(self,DF):
        col_list = DF.columns.values  # 列名を取得
        row = []    
        for col in col_list:
            tmp = ( col,  # 列名
                    DF[col].dtypes,  # データタイプ 
                    DF[col].isnull().sum(),  # null数 
                    DF[col].count(),  # データ数 (欠損値除く)
                    DF[col].nunique(),  # ユニーク値の数 (欠損値除く) 
                    DF[col].unique() )  # ユニーク値        
            row.append(tmp)  # tmpを順次rowに保存
        DF = pd.DataFrame(row)  # rowをデータフレームの形式に変換
        DF.columns = ['feature', 'dtypes', 'nan',  'count', 'num_unique', 'unique']  # データフレームの列名指定
        return DF


# In[ ]:




