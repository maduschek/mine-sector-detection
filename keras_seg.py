import os
import time
from datetime import datetime
import pdb
import platform
# from IPython.display import display
from tensorflow.keras.preprocessing.image import load_img
from PIL import ImageOps, Image
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers
from imgrender import render

# set paths
# cats dogs

'''
base_dir = "../../data/cats_dogs/"
input_dir_train = base_dir + "images/"
target_dir_train = base_dir + "annotations/trimaps"
input_dir_test = base_dir + "images/"
target_dir_test = base_dir + "annotations/trimaps"
img_size = (160, 160)
num_classes = 3
batch_size = 32
'''


# mine sectors
if platform.system() == "Windows":
    base_dir = "C:/data/mine-sectors/"
else:
    base_dir = "../../data/mine-sectors/"

input_dir_train = base_dir + "train_img/"
target_dir_train = base_dir + "train_seg/"
input_dir_test = base_dir + "test_img/"
target_dir_test = base_dir + "test_seg/"
img_size = (128, 128)
num_classes = 10
batch_size = 2


# load the image files
input_img_paths_train = sorted(
    [
        os.path.join(input_dir_train, fname)
        for fname in os.listdir(input_dir_train)
        if fname.endswith(".png")
    ]
)

# load the mask files
target_img_paths_train = sorted(
    [
        os.path.join(target_dir_train, fname)
        for fname in os.listdir(target_dir_train)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

# load the image files
input_img_paths_test = sorted(
    [
        os.path.join(input_dir_train, fname)
        for fname in os.listdir(input_dir_train)
        if fname.endswith(".png")
    ]
)

# load the mask files
target_img_paths_test = sorted(
    [
        os.path.join(target_dir_train, fname)
        for fname in os.listdir(target_dir_train)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

print("Number of samples:", len(input_img_paths_train))

# show image and its mask
for input_path, target_path in zip(input_img_paths_train[:], target_img_paths_train[:]):
    arr = np.array(Image.open(target_path))

    # the following step is necessarey becaus all 0 valued mask-pixels are set to 255
    if arr.min() == 0:
        print(target_path, " mask pixels +1")
        Image.fromarray(arr + 1).save(target_path)

    # if arr.max() > 3 or len(arr.shape) > :
    #     print(input_path, "|", target_path, " max: ", str(arr.max()))


if False:
    rnd_idx = (np.random.random(10)*len(input_img_paths_train)).astype(int)
    for i in rnd_idx:
        os.system('clear')
        render(os.path.join(input_img_paths_train[i]), scale=(128, 128))
        print(input_img_paths_train[i])
        print("---------------------------------------")

        mask = Image.open(os.path.join(target_img_paths_train[i]))
        ImageOps.autocontrast(mask).save(
            os.path.join("./", "mask_contrast" + str(i) + ".png"))
        render(os.path.join("./", "mask_contrast" + str(i) + ".png"), scale=(128, 128))
        print(target_img_paths_train[i])
        time.sleep(5)



class MineSectorHelper(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        return x, y


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


if __name__ == "__main__":
    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Build model
    model = get_model(img_size, num_classes)
    model.summary()

    """
    ## Set aside a validation split
    """

    import random

    # Split our img paths into a training and a validation set
    train_input_img_paths = input_img_paths_train
    train_target_img_paths = target_img_paths_train

    val_input_img_paths = input_img_paths_test
    val_target_img_paths = target_img_paths_test

    # Instantiate data Sequences for each split
    train_gen = MineSectorHelper(
        batch_size, img_size, train_input_img_paths, train_target_img_paths
    )
    val_gen = MineSectorHelper(batch_size, img_size, val_input_img_paths, val_target_img_paths)

    """
    ## Train the model
    """

    # Configure the model for training.
    # We use the "sparse" version of categorical_crossentropy
    # because our target data is integers.
    adam_opt = keras.optimizers.Adam(
        learning_rate=0.000005,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False
    )
    model.compile(optimizer=adam_opt, loss="sparse_categorical_crossentropy")

    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    callbacks = [
        keras.callbacks.ModelCheckpoint("mining-segments-model.h5", save_best_only=True),
        tensorboard_callback
    ]

    # Train the model, doing validation at the end of each epoch.
    epochs = 2
    model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
    input("Press Enter to continue...")

    """
    ## Visualize predictions
    """

    # Generate predictions for all images in the validation set

    val_gen = MineSectorHelper(batch_size, img_size, val_input_img_paths, val_target_img_paths)
    val_preds = model.predict(val_gen)

    print(val_preds.shape)



    def display_img_mask_gt(i):
        os.makedirs("images", exist_ok=True)

        pred_mask_path = os.path.join("./", "images", "mask" + str(i) + ".png")
        gt_mask_path = os.path.join("./", "images", "mask_gt" + str(i) + ".png")
        img_path = os.path.join("./", "images", "img" + str(i) + ".png")

        # predicted mask
        pred_mask = np.argmax(val_preds[i], axis=-1)
        pred_mask = np.expand_dims(pred_mask, axis=-1)
        ImageOps.autocontrast(keras.preprocessing.image.array_to_img(pred_mask)).\
            save(pred_mask_path)

        # ground truth mask
        gt_mask = np.asarray(Image.open(val_target_img_paths[i]))
        gt_mask = np.expand_dims(gt_mask, axis=-1)
        ImageOps.autocontrast(keras.preprocessing.image.array_to_img(gt_mask)). \
            save(gt_mask_path)

        # input image
        Image.open(val_input_img_paths[i]).\
            save(img_path)

        os.system('clear')
        render(pred_mask_path)
        print("---------------")
        render(gt_mask_path)
        print("---------------")
        render(img_path)
        time.sleep(5)

    # for i in range(10): display_img_mask_gt(i)
