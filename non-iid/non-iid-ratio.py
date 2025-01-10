import os
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

# 로컬 데이터 경로 및 저장 경로 설정
original_dataset_dir = '..'
non_iid_dataset_dir = '..'

# 클래스별 데이터 비율 설정
case_distributions = [
        [0.7, 0.1, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01],  # Client 1 heavily biased to class 0
        [0.05, 0.6, 0.15, 0.05, 0.05, 0.03, 0.03, 0.01, 0.01, 0.02],  # Client 2 heavily biased to class 1
        [0.1, 0.1, 0.4, 0.2, 0.1, 0.05, 0.01, 0.01, 0.01, 0.01],       # Client 3 heavily biased to class 2
        [0.05, 0.1, 0.05, 0.5, 0.1, 0.1, 0.05, 0.02, 0.01, 0.02],      # Client 4 heavily biased to class 3
        [0.1, 0.05, 0.05, 0.05, 0.5, 0.1, 0.05, 0.05, 0.03, 0.02],     # Client 5 heavily biased to class 4
]

# 클래스별 데이터 파일 수집
def collect_data_by_class(dataset_dir):
    data_by_class = {}
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_path):
            data_by_class[class_name] = [os.path.join(class_path, f) for f in os.listdir(class_path)]
    return data_by_class

# 데이터셋을 non-IID 형태로 분할 후 저장
def split_and_save_dataset(data_by_class, distributions, save_dir):
    for client_idx, distribution in enumerate(distributions):
        client_dir = os.path.join(save_dir, f'client_{client_idx + 1}')
        train_dir = os.path.join(client_dir, 'train')
        test_dir = os.path.join(client_dir, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        for class_idx, (class_name, files) in enumerate(data_by_class.items()):
            split_size = int(len(files) * distribution[class_idx])
            selected_files = files[:split_size]
            files = files[split_size:]  # 선택된 파일 이후의 파일 업데이트
            
            # train-test split
            train_files, test_files = train_test_split(selected_files, test_size=0.2, random_state=42)
            
            # 클라이언트 폴더에 해당 클래스의 데이터 저장
            class_train_dir = os.path.join(train_dir, class_name)
            class_test_dir = os.path.join(test_dir, class_name)
            os.makedirs(class_train_dir, exist_ok=True)
            os.makedirs(class_test_dir, exist_ok=True)
            
            for file_path in train_files:
                shutil.copy(file_path, class_train_dir)
            for file_path in test_files:
                shutil.copy(file_path, class_test_dir)

# 데이터 불러오기 및 분할하여 저장
if __name__ == "__main__":
    data_by_class = collect_data_by_class(original_dataset_dir)
    split_and_save_dataset(data_by_class, case_distributions, non_iid_dataset_dir)
    print("Non-IID 데이터셋 생성 완료!")
