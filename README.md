## 연합학습의 Non-IID 문제 개선을 위한 생성모델 적용 방안에 관한 연구


### 개발 기간
+ 2024.09 ~ 2024.11


### 파일 구성
+ federated_learning_partition
    - server.py
    - client.py
    - evaluation.py
      
+ federated_learning
    - server.py
    - client.py
    - evaluation.py

+ vae
    - vae_generation.py

+ non-iid
    -  non-iid-ratio.py

+ dir_set
    - distribute_data.py 
    - merge_dir.py


------------------------------------------------


### 연합학습 구현하기

windows 환경, anaconda에서 가상환경 구성 후 프롬프트에서 실행
server.py 파일 실행 후 client.py 파일 실행


    #server.py 실행
    python server.py

    #federated_learning_partition 실행
    python client.py --partition 0 --port 8080
    python client.py --partition 1 --port 8080
    python client.py --partition 2 --port 8080
    python client.py --partition 3 --port 8080
    python client.py --partition 4 --port 8080

    #federated_learning 실행
    python client.py --training_dir 테스트 데이터 파일 경로 A --test_dir  훈련 데이터 파일 경로 A --port 8080
    python client.py --training_dir 테스트 데이터 파일 경로 B --test_dir  훈련 데이터 파일 경로 B --port 8080
    .....


