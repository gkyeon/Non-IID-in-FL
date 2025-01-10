import os
import numpy as np
import flwr as fl
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import argparse

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument('--training_dir', type=str, required=True, help='Training data directory for the client')
    parser.add_argument('--test_dir', type=str, required=True, help='Test data directory for the client')
    parser.add_argument('--port', type=int, default=8080, help='Port number for the client')
    return parser.parse_args()

# CNN 모델 생성
def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# 로컬 JPG 이미지 데이터를 불러오는 함수
def load_data_from_jpg(training_dir, test_dir, img_size=(28, 28)): #이미지 사이즈 설정
    # 학습 데이터 로드
    image_paths = []
    labels = []
    for label_dir in os.listdir(training_dir):
        label_path = os.path.join(training_dir, label_dir)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                image_paths.append(img_path)
                labels.append(int(label_dir))  # 레이블 디렉토리 이름을 숫자로 사용

    # 이미지를 로드하고 전처리 (JPG -> 배열)
    images = []
    for img_path in image_paths:
        img = load_img(img_path, target_size=img_size, color_mode='grayscale')
        img_array = img_to_array(img).astype('float32') / 255.0  # 정규화
        images.append(img_array)

    x_train = np.array(images)
    y_train = np.array(labels)

    # 테스트 데이터 로드
    test_images = []
    test_labels = []
    for label_dir in os.listdir(test_dir):
        label_path = os.path.join(test_dir, label_dir)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                test_images.append(img_to_array(load_img(img_path, target_size=img_size, color_mode='grayscale')).astype('float32') / 255.0)
                test_labels.append(int(label_dir))

    x_test = np.array(test_images)
    y_test = np.array(test_labels)

    return x_train, y_train, x_test, y_test

# 연합학습 클라이언트 정의
class CNNClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.optimizer = Adam(1e-4)

    def get_parameters(self, config=None):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=128)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}

# 메인 함수
def main():
    args = parse_args()
    # 클라이언트별 데이터 경로를 기반으로 데이터 로드
    x_train, y_train, x_test, y_test = load_data_from_jpg(args.training_dir, args.test_dir)
    
    # CNN 모델 생성
    cnn_model = create_cnn_model()
    client = CNNClient(cnn_model, x_train, y_train, x_test, y_test)
    
    # Flower 클라이언트 시작
    fl.client.start_client(
        server_address=f"127.0.0.1:{args.port}",
        client=client.to_client()  # NumPyClient 객체를 Client로 변환하여 반환
    )

if __name__ == "__main__":
    main()
