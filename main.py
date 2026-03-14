from scripts.unsupervised import run_unsupervised
from scripts.supervised import run_supervised
from scripts.cnn import run_cnn


def main():

    print("Starting ML pipeline")

    run_unsupervised()
    run_supervised()
    run_cnn()

    print("Pipeline complete")


if __name__ == "__main__":
    main()