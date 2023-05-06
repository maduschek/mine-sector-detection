import random
import glob
import os
import time
from datetime import datetime
import pdb
import platform
# from IPython.display import display
from tensorflow.keras.preprocessing.image import load_img
from PIL import ImageOps, Image
from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers
from imgrender import render
from sklearn.metrics import confusion_matrix
np.set_printoptions(edgeitems=30, linewidth=100000,
    formatter=dict(float=lambda x: "%.3g" % x))


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

# a priori knowledge
# class weights gained from seg_stats.py
class_weights = {0: 1,
                 1: 50.3606,
                 2: 195.0304,
                 3: 202.9463,
                 4: 64.3299,
                 5: 83.5509,
                 6: 745.9514,
                 7: 119.1049,
                 8: 41.1413,
                 9: 33.946,
                 10: 33216.4873}

# mine sectors
if platform.system() == "Windows":
    base_dir = "C:/data/mine-sectors/"
else:
    base_dir = "/home/maduschek/ssd/mine-sector-detection/"
    # base_dir = "/home/maduschek/data/cats_dogs/"

input_dir_train = base_dir + "images_trainset/"
target_dir_train = base_dir + "masks_trainset/"
input_dir_test = base_dir + "images_testset/"
target_dir_test = base_dir + "masks_testset/"
img_size = (256, 256)
num_classes = 10
batch_size = 16

epochs = int(input("epochs: "))
subset_percent = float(input("subset_size in %: "))/100


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
        os.path.join(input_dir_test, fname)
        for fname in os.listdir(input_dir_test)
        if fname.endswith(".png")
    ]
)

# load the mask files
target_img_paths_test = sorted(
    [
        os.path.join(target_dir_test, fname)
        for fname in os.listdir(target_dir_test)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)


# random subset for faster DEBUGGING
subset_idx_train = (np.random.random(int(len(input_img_paths_train) * subset_percent)) * len(input_img_paths_train)).astype(int)
input_img_paths_train = np.asarray(input_img_paths_train)[subset_idx_train]
target_img_paths_train = np.asarray(target_img_paths_train)[subset_idx_train]

subset_idx_test = (np.random.random(int(len(input_img_paths_test) * subset_percent)) * len(input_img_paths_test)).astype(int)
input_img_paths_test = np.asarray(input_img_paths_test)[subset_idx_test]
target_img_paths_test = np.asarray(target_img_paths_test)[subset_idx_test]

print("Number of train samples:", len(input_img_paths_train))
print("Number of test samples:", len(input_img_paths_test))



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
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")

        for j, path in enumerate(batch_input_img_paths):
            # print(path)
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")

        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        return x, y


def get_item(idx):
    """Returns tuple (input, target) correspond to batch #idx."""
    i = idx * self.batch_size
    batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
    batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]
    x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")

    for j, path in enumerate(batch_input_img_paths):
        # print(path)
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


def get_optimizer(optimizer="adam"):

    if optimizer == "adam":
        return keras.optimizers.Adam(
            learning_rate=0.00005,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False)

    if optimizer == "sgd":
        return keras.optimizers.SGD(
            learning_rate=0.01,
            momentum=0.0,
            nesterov=False,
            weight_decay=None)

    if optimizer == "rmsprop":
        return keras.optimizers.RMSprop(
            learning_rate=0.000000001,
            rho=0.9,
            momentum=0.0,
            epsilon=1e-07,
            centered=False,
            weight_decay=None)

    if optimizer == "adagrad":
        return keras.optimizers.Adagrad(
            learning_rate=0.001,
            initial_accumulator_value=0.1,
            epsilon=1e-07,
            weight_decay=None)


if __name__ == "__main__":
    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Build model
    model = get_model(img_size, num_classes)
    model.summary()

    """
    ## Set aside a validation split
    """

    # Split our img paths into a training and a validation set
    train_input_img_paths = input_img_paths_train
    train_target_img_paths = target_img_paths_train

    val_input_img_paths = input_img_paths_test
    val_target_img_paths = target_img_paths_test

    print(val_input_img_paths[0])
    print(train_input_img_paths[0])

    # Instantiate data Sequences for each split
    train_gen = MineSectorHelper(batch_size, img_size, train_input_img_paths, train_target_img_paths)
    val_gen = MineSectorHelper(batch_size, img_size, val_input_img_paths, val_target_img_paths)


    # load the model
    model_file = "mining-segments-model.h5"
    if os.path.isfile(model_file):
        model = keras.models.load_model("mining-segments-model.h5")
    else:

        # Train the model
        # Configure the model for training.
        # We use the "sparse" version of categorical_crossentropy
        # because our target data is integers.

        model.compile(optimizer=get_optimizer("rmsprop"), loss="categorical_crossentropy")
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        callbacks = [
            keras.callbacks.ModelCheckpoint("mining-segments-model.h5", save_best_only=True),
            tensorboard_callback]

        # Train the model, doing validation at the end of each epoch.
        model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)


    # Visualize predictions

    # Generate predictions for all images in the validation set
    # val_gen = MineSectorHelper(batch_size, img_size, val_input_img_paths, val_target_img_paths)
    # val_preds = model.predict(val_gen, workers=1, max_queue_size=2)

    sum_conf_mat = np.array([])
    acc_total = 0
    count = 0
    for img_path, mask_path in zip(val_input_img_paths, val_target_img_paths):
        count += 1
        img_arr = np.asarray(Image.open(img_path))
        mask_arr = np.asarray(Image.open(mask_path))

        dict_metrics = model.evaluate(x=np.expand_dims(img_arr, axis=0), y=np.expand_dims(mask_arr, axis=0), return_dict=True)
        pred_mask = model.predict(np.expand_dims(img_arr, axis=0))

        # save the predicted mask file and compare with gt target
        os.makedirs("output", exist_ok=True)
        pred_mask_path = os.path.join("./", "output", "mask_pred" + os.path.basename(img_path) + ".png")
        pred_mask = np.argmax(pred_mask, axis=-1)
        # pred_mask = np.expand_dims(pred_mask[0], axis=-1)

        Image.fromarray(((pred_mask/10)*255).astype('B')[0]).save(pred_mask_path)

        os.system('clear')
        # render(img_path)
        # print("-------------")
        # render(mask_path)
        # print("-------------")
        # render(pred_mask_path)

        res = (mask_arr == pred_mask[0]).astype(int)
        acc = np.sum(res)/np.size(res)
        acc_total += acc
        print("total accuracy: ", acc_total / count)

        if sum_conf_mat.size == 0:
            sum_conf_mat = confusion_matrix(y_true=mask_arr.flatten(), y_pred=pred_mask[0].flatten(), labels=range(10))
        else:
            conf_mat = confusion_matrix(y_true=mask_arr.flatten(), y_pred=pred_mask[0].flatten(), labels=range(10))
            sum_conf_mat += conf_mat
            print(sum_conf_mat)

    plt.matshow(sum_conf_mat)

    a = input("continue...")






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
