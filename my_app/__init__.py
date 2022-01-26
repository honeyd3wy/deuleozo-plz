import joblib
import flask
import sqlite3
import pandas as pd
import sklearn.externals
from pathlib import Path
from flask import Flask, request, render_template
from flask_restful import Resource, Api


import sys
#sys.path.append('C:/Users/ylwhd/section-3-project/my_app')

import pr3mylibrary1.get_myplaylist_api as pl
import pr3mylibrary2.data_handling as dh
from pr3mylibrary3.model import export_model

app = Flask(__name__)
api = Api(app)

#메인 페이지
@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')

# 컬렉션에 저장할 정보 입력받는 창
@app.route("/add-playlist", methods=['GET'])
def save_playlist():
    if request.method == 'GET':
        return render_template('upload_data.html')

# 컬렉션에 playlist 저장
@app.route("/add-playlist/complete", methods=['POST'])
def complete():
    if request.method == 'POST':
        dir = str(Path(__file__).parent)

        your_name = str(request.form['your_name'])
        username = str(request.form['username'])
        playlist_id = str(request.form['playlist_id'])
        liked = int(request.form['liked'])
        status = '플레이리스트 업로드'

        dh.SaveToMongoDB(your_name=your_name, username=username, playlist_id=playlist_id, liked=liked).save_to_mongo()

        df_merge = dh.MergeMongoDB().merge_all_data(dir=dir)

        # DB에 저장
        df_sql = dh.transform_df(df_merge)
        conn = sqlite3.connect(f'{dir}/total_userdata/total.db')
        df_sql.to_sql(f'{dir}/total_userdata/total.db', conn, if_exists='replace')
        conn.close()
        return render_template("complete.html", status=status)

# 모델링을 위한 유저 이름을 받는 창
@app.route('/modeling', methods=['GET'])
def get_id_for_model():
    if request.method == 'GET':
        return render_template('model_getID.html')

# 사용자 이름을 받아 모델링
@app.route("/modeling/complete", methods=['POST'])
def modeling():
    if request.method == 'POST':
        dir = str(Path(__file__).parent.parent)
        your_name = str(request.form['your_name'])
        status = '추천 모델 학습 및 생성'

        df = dh.GetFromMongoDB().get_from_mongo(your_name)
        # 모델링
        export_model(dir=dir, dataframe=df, your_name=your_name, mode=None)

        return render_template("complete.html", status=status), 201
    else:
        return 'ERROR', 404


# 예측 받는 창
@app.route('/recommendation', methods=['GET'])
def get_id_for_rec():
    if request.method == 'GET':
        return render_template('rec_getID.html')

#  예측 결과
@app.route('/recommendation/complete', methods=['POST'])
def recommentation():
    
    # 쿼리 파라미터 받기
    your_name = str(request.form['your_name'])

    features_col = dh.Features().featrues_col()
    dir = str(Path(__file__).parent)

    # my_playlist 불러오기
    _, my_df = dh.MyTracksInfoList(username='jr', playlist_id='4Csbv2gvxH3AdrnLbEJfI5', liked=0).get_info()
    # 모델 불러오기
    model = joblib.load(dir + f'/rf_model/rcmd_{your_name}.pkl')
    predict = model.predict(my_df[features_col])
    predict_proba = model.predict_proba(my_df[features_col])[:,1]

    my_df['predict'] = predict
    my_df['predict_proba'] = predict_proba
    my_df.drop('liked', axis=1, inplace=True)
    rec_df = my_df[my_df['predict_proba'] > 0.6]

    rec_result = pl.SongArtist().get_song_artist(rec_df['id'])

    return render_template('rec_result.html', your_name=your_name, song_artist_lst=rec_result)

# 재학습을 위해 id 받는 창
@app.route('/retrain', methods=['GET'])
def get_id_for_retrain():
    if request.method == 'GET':
        return render_template('rt_getID.html')

# 데이터 모델 재학습
@app.route('/retrain/result', methods=['POST'])
def make_model():
    if request.method == 'POST':
        dir = str(Path(__file__).parent.parent)
        your_name = str(request.form['your_name'])
        status = '모델 재생성'
        df = dh.GetFromMongoDB().get_from_mongo(your_name)
        # 모델 재 생성(mode='R')
        export_model(dir=dir, dataframe=df, your_name=your_name, mode='R')
        return render_template('complete.html', status=status)

# 콜렉션 삭제를 위해 id 받는 창
@app.route('/delete', methods=['GET'])
def get_id_for_del():
    if request.method == 'GET':

        return render_template('del_getID.html')

# 콜렉션 삭제
@app.route('/delete/complete', methods=['POST'])
def delete_col():
    if request.method == 'POST':
        dir = str(Path(__file__).parent)
        your_name = str(request.form['your_name'])
        status = '데이터 삭제'
        dh.DeleteUser().delete_yourname(your_name=your_name)

        df_merge = dh.MergeMongoDB().merge_all_data(dir=dir)
        # DB에 저장
        df_sql = dh.transform_df(df_merge)
        conn = sqlite3.connect(f'{dir}/total_userdata/.db')
        df_sql.to_sql(f'{dir}/total_userdata/total_userdata.db', conn, if_exists='replace')
        conn.close()
        return render_template('complete.html', status=status)


if __name__ == "__main__":
    # 모델 로드
    app.run(host='0.0.0.0', port=8000, debug=True)
