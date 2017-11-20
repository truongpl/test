# [ Truong] This script used to load the already trained model
from utilities.dataUtils import CoNLLDataset
from engine.ner import NERModel
from utilities.config import Config


def main():
    config = Config()

    # build model
    model = NERModel(config)
    model.buildGraph()
    model.restoreSession('results/test/model.weights/')
    # model.reinitialize_weights("proj")

if __name__ == "__main__":
    main()
