import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras import layers, models, callbacks, losses, optimizers, metrics

# 이미지 표시 함수
def display(images, n=10):
    plt.figure(figsize=(20, 2))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(np.squeeze(images[i]), cmap="gray")
        plt.axis("off")
    plt.show()

# 데이터 전처리 함수
def preprocess(imgs):
    imgs = imgs.astype("float32") / 255.0
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
    imgs = np.expand_dims(imgs, -1)
    return imgs

# 데이터 불러오기
def load_local_images(directory, target_size=(32, 32)):
    images = []
    labels = []
    class_names = os.listdir(directory)
    
    for label in class_names:
        class_dir = os.path.join(directory, label)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            img = Image.open(file_path).resize(target_size).convert("L")
            img = np.array(img) / 255.0
            images.append(img)
            labels.append(int(label))
    
    return np.array(images), np.array(labels)

# 생성된 이미지를 저장하는 함수
def save_generated_images(images, save_path):
    os.makedirs(save_path, exist_ok=True)
    for i, image in enumerate(images):
        img = Image.fromarray((image.squeeze() * 255).astype(np.uint8))
        img.save(os.path.join(save_path, f"generated_image_{i}.png"))

# Sampling 레이어 정의
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# VAE 인코더 정의
IMAGE_SIZE = 32
EMBEDDING_DIM = 100
BETA = 500

encoder_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), name="encoder_input")
x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(encoder_input)
x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)
shape_before_flattening = K.int_shape(x)[1:]

x = layers.Flatten()(x)
z_mean = layers.Dense(EMBEDDING_DIM, name="z_mean")(x)
z_log_var = layers.Dense(EMBEDDING_DIM, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")

# VAE 디코더 정의
decoder_input = layers.Input(shape=(EMBEDDING_DIM,), name="decoder_input")
x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
x = layers.Reshape(shape_before_flattening)(x)
x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
decoder_output = layers.Conv2DTranspose(1, (3, 3), activation="sigmoid", padding="same", name="decoder_output")(x)

decoder = models.Model(decoder_input, decoder_output)

# VAE 모델 정의
class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    def train_step(self, data):
        # 데이터를 reconstruction과 같은 형태로 변환
        data = tf.expand_dims(data, axis=-1)

        with tf.GradientTape() as tape:
            z_mean, z_log_var, reconstruction = self(data)
            reconstruction_loss = tf.reduce_mean(
                BETA * losses.binary_crossentropy(data, reconstruction)
            )
            kl_loss = tf.reduce_mean(
                tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

# 나머지 코드 동일


# VAE 모델 생성
vae = VAE(encoder, decoder)
vae.compile(optimizer=optimizers.Adam())

# 로컬에서 데이터 불러오기
local_data_dir = ".."
x_train, y_train = load_local_images(local_data_dir)

# 모델 훈련
#vae.fit(x_train, epochs=100, batch_size=64)

# 잠재 공간에서 새로운 이미지 생성
def generate_new_images(decoder, num_images=10):
    random_latent_vectors = np.random.normal(size=(num_images, EMBEDDING_DIM))
    generated_images = decoder.predict(random_latent_vectors)
    return generated_images

# 레이블별 잠재 공간에서 샘플링하여 이미지를 생성하고 저장하는 함수
def generate_and_save_by_label(vae, labels, num_images_per_label=6000):
    """
    각 레이블의 잠재 공간에서 이미지를 생성하고 레이블별 디렉토리에 저장합니다.
    """
    save_path = "./generated_images_by_label"
    os.makedirs(save_path, exist_ok=True)

    unique_labels = np.unique(labels)  # 각 고유 레이블에 대해

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        z_mean, z_log_var, _ = vae.encoder.predict(x_train[label_indices])  # 해당 레이블 데이터의 잠재 공간
        
        # 각 레이블의 평균 잠재 벡터에서 생성된 이미지를 저장
        for i in range(num_images_per_label):
            epsilon = np.random.normal(size=z_mean.shape[1])
            sampled_z = z_mean.mean(axis=0) + np.exp(0.5 * z_log_var.mean(axis=0)) * epsilon
            generated_image = vae.decoder.predict(np.array([sampled_z]))
            
            # 이미지 저장 경로 설정
            label_dir = os.path.join(save_path, str(label))
            os.makedirs(label_dir, exist_ok=True)
            
            img = Image.fromarray((generated_image.squeeze() * 255).astype(np.uint8))
            img.save(os.path.join(label_dir, f"generated_{i}.png"))

# VAE 모델 훈련 후 생성된 이미지 저장 호출
vae.fit(x_train, epochs=100, batch_size=64, shuffle=True)
generate_and_save_by_label(vae, y_train, num_images_per_label=6000)










# 생성된 이미지 생성 및 저장
#new_images = generate_new_images(decoder, num_images=10)
#save_generated_images(new_images, "./generated_images_fashion_mnist")

print("새로운 생성된 이미지를 './generated_images_cifar10' 폴더에 저장했습니다.")
