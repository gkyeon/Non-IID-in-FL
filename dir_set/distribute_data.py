import os
import random
import shutil
from collections import defaultdict

def distribute_data_with_replacement(source_dir, output_dir, case_distributions, classes, max_train=6000, max_test=600):
    """
    데이터를 랜덤으로 선택하여 클래스별, 클라이언트별 비율에 따라 분배하며,
    데이터가 부족할 경우 중복을 허용합니다.

    :param source_dir: 원본 데이터셋 디렉토리 (data)
    :param output_dir: 분배된 데이터셋 디렉토리
    :param case_distributions: 클래스별 비율 설정
    :param classes: 클래스 리스트 (예: [0, 1, ..., 9])
    :param max_train: train 데이터에서 클래스별 최대 개수
    :param max_test: test 데이터에서 클래스별 최대 개수
    """
    for partition, max_files in [('train', max_train), ('test', max_test)]:
        print(f"Processing {partition} data...")

        # 클래스별 이미지 파일 저장
        class_files = defaultdict(list)
        partition_path = os.path.join(source_dir, partition)

        # 원본 데이터 클래스별 파일 수집
        for cls in classes:
            cls_path = os.path.join(partition_path, str(cls))
            if os.path.exists(cls_path):
                for img_file in os.listdir(cls_path):
                    class_files[cls].append(os.path.join(cls_path, img_file))

        # 클래스별 파일 셔플
        for cls in classes:
            random.shuffle(class_files[cls])

        # 클라이언트별 데이터 분배
        for cls in classes:
            total_files = len(class_files[cls])
            distributed_count = [0] * len(case_distributions)  # 클라이언트별 분배 개수 추적

            for client_id, ratio in enumerate([d[cls] for d in case_distributions]):
                num_files = int(max_files * ratio)  # 비율에 따른 파일 수
                client_dir = os.path.join(output_dir, f"client{client_id + 1}/{partition}/{cls}")
                os.makedirs(client_dir, exist_ok=True)

                # 중복 허용 랜덤 선택
                selected_files = [random.choice(class_files[cls]) for _ in range(num_files)]
                for file in selected_files:
                    target_file = os.path.join(client_dir, os.path.basename(file))
                    shutil.copy(file, target_file)

                # 분배된 개수 기록
                distributed_count[client_id] += len(selected_files)

            # 확인 로그
            print(f"Class {cls}: Distributed {distributed_count} files among clients")

case_distributions = [
    [0.3, 0.9, 0.95, 0.95, 0.95, 0.99, 0.99, 0.99, 0.99, 0.99],  # Client 1 less biased to class 0
    [0.95, 0.4, 0.85, 0.95, 0.95, 0.97, 0.97, 0.99, 0.99, 0.98],  # Client 2 less biased to class 1
    [0.9, 0.9, 0.6, 0.8, 0.9, 0.95, 0.99, 0.99, 0.99, 0.99],       # Client 3 less biased to class 2
    [0.95, 0.9, 0.95, 0.5, 0.9, 0.9, 0.95, 0.98, 0.99, 0.98],      # Client 4 less biased to class 3
    [0.9, 0.95, 0.95, 0.95, 0.5, 0.9, 0.95, 0.95, 0.97, 0.98],     # Client 5 less biased to class 4
]

# 실행
source_dir = ".."  # 원본 데이터셋 디렉토리
output_dir = ".."  # 분배된 데이터셋 디렉토리
classes = list(range(10))  # 클래스 리스트
distribute_data_with_replacement(source_dir, output_dir, case_distributions, classes, max_train=6000, max_test=600)


