
from read import read_sentences
from data_util import create_dataset,save_dataset

def main():
    dataset_save_location = "data/data.p"

    X, Y = read_sentences("data/hindencorp05.plaintext")
    save_dataset(dataset_save_location, create_dataset(X, Y))


if __name__ == '__main__':
    main()
