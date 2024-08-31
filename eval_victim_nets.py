from nets import load_network
import os
import wget
import gzip
import tarfile
import pickle
import numpy as np
import matplotlib.pyplot as plt

# mnist_train_images_url = "https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
# mnist_train_labels_url = "https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
# mnist_test_images_url = "https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
# mnist_test_labels_url = "https://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

mnist_train_images_url = "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"
mnist_train_labels_url = "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz"
mnist_test_images_url = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"
mnist_test_labels_url = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"

cifar10_dataset_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

def un_gz(file_path:str):
    g_file = gzip.GzipFile(file_path)
    result_path = file_path.replace(".gz", "")
    with open(result_path, "wb") as f:
        f.write(g_file.read())
    g_file.close()

def un_tar(file_path:str):
    tar = tarfile.open(file_path)
    result_folder = file_path.replace(".tar", "/")
    names = tar.getnames()
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)
    for name in names:
        tar.extract(name, result_folder)
    tar.close()

def convert_bytes_to_int(bytes_array:bytes, endian="big_endian"):
    res = 0
    if endian == "big_endian":
        iterator = bytes_array
    else:
        iterator = reversed(bytes_array)
    for i in iterator:
        res = (res << 8) | i
    return res

def download_mnist_dataset(save_folder:str):
    global mnist_train_images_url, mnist_train_labels_url, mnist_test_images_url, mnist_test_labels_url
    if not (save_folder.endswith("/") or save_folder.endswith("\\")):
        save_folder += "/"
    train_images_path = save_folder + "train-images-idx3-ubyte"
    train_labels_path = save_folder + "train-labels-idx1-ubyte"
    test_images_path = save_folder + "t10k-images-idx3-ubyte"
    test_labels_path = save_folder + "t10k-labels-idx1-ubyte"
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    # download and ungz
    if not os.path.exists(train_images_path):
        print("\ndownloading train-images-idx3-ubyte.gz file...")
        wget.download(mnist_train_images_url, train_images_path + ".gz")
        un_gz(train_images_path + ".gz")
    if not os.path.exists(train_labels_path):
        print("\ndownloading train-labels-idx1-ubyte.gz file...")
        wget.download(mnist_train_labels_url, train_labels_path + ".gz")
        un_gz(train_labels_path + ".gz")
    if not os.path.exists(test_images_path):
        print("\ndownloading t10k-images-idx3-ubyte.gz file...")
        wget.download(mnist_test_images_url, test_images_path + ".gz")
        un_gz(test_images_path + ".gz")
    if not os.path.exists(test_labels_path):
        print("\ndownloading t10k-labels-idx1-ubyte.gz file...")
        wget.download(mnist_test_labels_url, test_labels_path + ".gz")
        un_gz(test_labels_path + ".gz")

def download_cifar10_dataset(save_folder:str):
    global cifar10_dataset_url
    if not (save_folder.endswith("/") or save_folder.endswith("\\")):
        save_folder += "/"
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    # download, ungz and untar
    file_path = save_folder + "cifar-10-python"
    if not os.path.exists(file_path):
        print("\ndownloading cifar-10-python.tar.gz file...")
        wget.download(cifar10_dataset_url, file_path + ".tar.gz")
        un_gz(file_path + ".tar.gz")
        un_tar(file_path + ".tar")

def load_labels_from_file_mnist(file_path):
    with open(file_path, mode="rb") as f:
        bytes_array = f.read()
    # check magic number first
    magic_number = convert_bytes_to_int(bytes_array[:4])
    assert magic_number == 0x801
    samples_number = convert_bytes_to_int(bytes_array[4:8])
    bytes_array = bytes_array[8:]
    assert len(bytes_array) == samples_number
    labels = np.array([i for i in bytes_array], dtype=np.uint8)
    return labels

def load_images_from_file_mnist(file_path):
    with open(file_path, mode="rb") as f:
        bytes_array = f.read()
    # check magic number first
    magic_number = convert_bytes_to_int(bytes_array[:4])
    assert magic_number == 0x803
    samples_number = convert_bytes_to_int(bytes_array[4:8])
    rows_number = convert_bytes_to_int(bytes_array[8:12])
    columns_number = convert_bytes_to_int(bytes_array[12:16])
    assert rows_number == 28 and columns_number == 28
    bytes_array = bytes_array[16:]
    assert len(bytes_array) == rows_number * columns_number * samples_number
    images = np.zeros((samples_number, rows_number, columns_number), dtype=np.uint8)
    byte_index = 0
    for i in range(samples_number):
        for j in range(rows_number):
            for k in range(columns_number):
                images[i, j, k] = bytes_array[byte_index]
                byte_index += 1
    return images

def load_mnist_data(dataset_folder:str):
    if not (dataset_folder.endswith("/") or dataset_folder.endswith("\\")):
        dataset_folder += "/"
    train_labels_path = dataset_folder + "train-labels-idx1-ubyte"
    train_images_path = dataset_folder + "train-images-idx3-ubyte"
    test_labels_path = dataset_folder + "t10k-labels-idx1-ubyte"
    test_images_path = dataset_folder + "t10k-images-idx3-ubyte"
    train_labels = load_labels_from_file_mnist(train_labels_path)
    train_images = load_images_from_file_mnist(train_images_path)
    test_labels = load_labels_from_file_mnist(test_labels_path)
    test_images = load_images_from_file_mnist(test_images_path)
    train_images = np.reshape(train_images, (len(train_images), -1))
    test_images = np.reshape(test_images, (len(test_images), -1))
    return (train_images, train_labels), (test_images, test_labels)

def load_cifar10_data(dataset_folder:str):
    if not (dataset_folder.endswith("/") or dataset_folder.endswith("\\")):
        dataset_folder += "/"
    data_batch_folder = dataset_folder + "cifar-10-python/cifar-10-batches-py/"
    train_images_batches = []
    train_labels_batches = []
    for batch_index in range(1, 6):
        with open(data_batch_folder + "data_batch_{}".format(batch_index), "rb") as f:
            dict = pickle.load(f, encoding="bytes")
        train_images_batches.append(dict[bytes("data", encoding="utf-8")])
        train_labels_batches.append(dict[bytes("labels", encoding="utf-8")])
    train_images = np.concatenate(train_images_batches, axis=0)
    train_labels = np.concatenate(train_labels_batches)
    with open(data_batch_folder + "test_batch", "rb") as f:
        dict = pickle.load(f, encoding="bytes")
        test_images = dict[bytes("data", encoding="utf-8")]
        test_labels = np.array(dict[bytes("labels", encoding="utf-8")], dtype=np.uint64)
    return (train_images, train_labels), (test_images, test_labels)

def download_and_load_dataset(data_set_name="mnist", data_folder="./datasets/"):
    '''
    Download and then load a dataset.
    
    :param data_set_name: dataset name, which can be 'mnist' or 'cifar10'
    :param data_folder: the folder in which the dataset will be saved
    :return: a dataset the sturcture of which is as ((train_images, train_labels), (test_images, test_labels))
    '''
    if not (data_folder.endswith("/") or data_folder.endswith("\\")):
        data_folder += "/"
    dataset_folder = data_folder + data_set_name + "/"
    if data_set_name == "mnist":
        download_mnist_dataset(dataset_folder)
        return load_mnist_data(dataset_folder)
    elif data_set_name == "cifar10":
        download_cifar10_dataset(dataset_folder)
        return load_cifar10_data(dataset_folder)

def choose_two_classes_from_dataset(dataset, p_number, n_number):
    '''
    Build a binary classification dataset from a dataset

    :param dataset: the overall dataset
    :param p_number: a number between 0 to 9, representing the class as positive class
    :param n_number: a number between 0 to 9, representing the class as negative class
    :return: a binary classification dataset, the structure of which is as ((train_images, train_labels), (test_images, test_labels))
    '''
    train_set, test_set = dataset
    train_p_images = train_set[0][train_set[1] == p_number, :]
    train_n_images = train_set[0][train_set[1] == n_number, :]
    train_images = np.concatenate((train_p_images, train_n_images), axis=0)
    train_labels = np.concatenate((np.ones(len(train_p_images), dtype=np.uint8), np.zeros(len(train_n_images), dtype=np.uint8)))
    # shuffle training set
    shuffled_indexes = np.random.permutation(len(train_images))
    train_images = train_images[shuffled_indexes, :]
    train_labels = train_labels[shuffled_indexes]
    test_p_images = test_set[0][test_set[1] == p_number, :]
    test_n_images = test_set[0][test_set[1] == n_number, :]
    test_images = np.concatenate((test_p_images, test_n_images), axis=0)
    test_labels = np.concatenate((np.ones(len(test_p_images), dtype=np.uint8), np.zeros(len(test_n_images), dtype=np.uint8)))
    return (train_images, train_labels), (test_images, test_labels)

def show_sample_images(image, data_set_name="mnist"):
    if data_set_name == "mnist":
        plt.figure("Example image of mnist dataset")
        plt.imshow(image.reshape(28,28))
        plt.show()
    elif data_set_name == "cifar10":
        plt.figure("Example image of cifar10 dataset")
        plt.imshow(image.reshape(3, 32, 32).transpose(1, 2, 0))
        plt.show()

if __name__ == "__main__":
    # choose test dataset
    # test_dataset = "mnist"
    test_dataset = "cifar10"

    if test_dataset == "mnist":
        # Load mnist dataset
        print("loading mnist dataset...")
        mnist_dataset = download_and_load_dataset("mnist", "./datasets/")

        # show an example image for each dataset
        print("\nshowing an example image for mnist dataset...")
        show_sample_images(mnist_dataset[0][0][0], "mnist")

        # test accuracy of victim models for mnist dataset
        print("\ntesting accuracy...")
        for i in range(5):
            p_class = 2 * i
            n_class = 2 * i + 1
            model = load_network("./models/mnist/mnist_{}vs{}_784_2_1.npz".format(p_class, n_class))
            (x_train, y_train), (x_test, y_test) = choose_two_classes_from_dataset(mnist_dataset, p_class, n_class)
            x_train, x_test = x_train / 255.0, x_test / 255.0
            print("Accuracy of mnist-{}vs{} model is {}.".format(p_class, n_class, model.accuracy(x_test, y_test)))
    elif test_dataset == "cifar10":
        # Load cifar10 dataset
        print("loading cifar10 dataset...")
        cifar10_dataset = download_and_load_dataset("cifar10", "./datasets/")

        # show an example image for each dataset
        print("\nshowing an example image for cifar10 dataset...")
        show_sample_images(cifar10_dataset[0][0][0], "cifar10")
    
        # test accuracy of victim models for cifar10 dataset
        print("\ntesting accuracy...")
        for i in range(5):
            p_class = 2 * i
            n_class = 2 * i + 1
            model = load_network("./models/cifar10/cifar10_{}vs{}_3072_2_1.npz".format(p_class, n_class))
            (x_train, y_train), (x_test, y_test) = choose_two_classes_from_dataset(cifar10_dataset, p_class, n_class)
            x_train, x_test = x_train / 255.0, x_test / 255.0
            print("Accuracy of cifar10-{}vs{} model is {}.".format(p_class, n_class, model.accuracy(x_test, y_test)))