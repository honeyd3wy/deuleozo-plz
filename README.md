# deuleozo-plz
_제가 좋아하는 노래는 어떠세요? 들어주세요....제발._

![](https://drive.google.com/uc?export=download&id=1JUZRMkaYuyxDW3p3ko-w9UCFODWKttTu)

- flask app 배포 연습용으로 만든 개인 토이 프로젝트입니다.

- 샘플 모델의 데이터는 이쪽에서 얻었습니다.

  : https://www.kaggle.com/bricevergnou/spotify-recommendation



## Info.


### I. What is this App?

![](https://drive.google.com/uc?export=download&id=14RYT284Xka0hxTGDxJgwmIVCu_ajortD)

### II. Data Pipeline

![](https://drive.google.com/uc?export=download&id=12Mee1uI2wl-iW5mMJEAag5-GwwYNgNNJ)

### III. 사용된 자체 제작 모듈

- API로 데이터 수집하는 코드 : https://github.com/honeyd3wy/pr3mylibrary1/blob/main/pr3mylibrary1/get_myplaylist_api.py
- 데이터를 다루는 코드 : https://github.com/honeyd3wy/pr3mylibrary2/blob/main/pr3mylibrary2/data_handling.py
- 랜덤포레스트 모델링 코드 : https://github.com/honeyd3wy/pr3mylibrary3/blob/main/pr3mylibrary3/model.py



## How to Use It?


### I. 접속 방법
1. CLI 창에 `$ git clone https://github.com/honeyd3wy/deuleozo-plz.git` 을 입력해 레포지토리를 클론합니다.

2. `$ cd /deuleozo-plz`으로 해당 디렉토리로 이동합니다.

3. `$ pip install -r requirements.txt`를 입력하면 필요한 패키지가 자동으로 설치됩니다.
 
4. `$ FLASK_APP=my_app flask run`을 입력하면 CLI 창에 링크가 뜹니다. ex) http://127.0.0.1:5000/

5. 해당 링크를 눌러주면 사용 준비 완료!



### II. 사용 방법
페이지에 설명이 굉장히 불친절합니다... 그래서 사용 방법을 첨부합니다.

1. 설정 및 관리 -> 플레이리스트 추가

2. 빈칸에 대한 설명은 다음과 같습니다. 토씨 하나만 틀려도 오류가 나니 주의해주세요.

>  - `Your Name` : 사용자 고유 닉네임(숫자, 영문, -, _ 만 넣어주세요.)을 입력합니다. 여기서만 사용되는 이름입니다.
>  
>  - `Playlist Username` : **플레이리스트를 만든 유저의 닉네임**을 넣어주세요. 
>  
>  - `Playlist ID` : **플레이리스트 고유 아이디**입니다.
>  
>  - `Do You Like it?` : 좋아하는 노래의 플레이리스트라면 Yes, 그렇지 않다면 No를 선택해주세요.
>  
>  Liked와 Disliked 플레이리스트가 최소 각각 하나씩은 입력되어야 추천 모델이 정상적으로 생성됩니다.
>  
>  시간이 조금 걸릴 수 있습니다. 인내심을 가지고 기다려주세요.

3. 설정 및 관리 -> 추천 모델 학습

4. 추천받기 -> 사용자 닉네임 입력

5. 조금만 기다리시면 결과가 나옵니다.

6. 플레이리스트를 업데이트 했다면 '모델 업데이트'로 모델도 업데이트 해주세요.


      #### 호불호 데이터가 균일하게 많이 누적될수록 정확도가 올라갑니다!


7. '서비스 그만 이용하기'에서 데이터를 삭제할 수 있습니다. 



사람마다 결과가 다른 이유는, 좋아할 확률이 65%가 넘는 노래들만 나오기 때문이에요.

노래가 너무 많이 나온다구요...? 그런 당신이 저의 취향메이트.



### III. 사용 종료

CLI 창에서 `Ctrl+c`를 누르시면 종료됩니다.

어떤 이유로든 ***flask run이 종료되면 더 이상 접속이 안 되는 점*** 참고하세요!
