# [Truong] This reference script used to train the LSTM 
from utilities.dataUtils import CoNLLDataset
from engine.ner import NERModel
from utilities.config import Config


def main():
    config = Config()

    # build model
    model = NERModel(config)
    model.buildGraph()

    # create datasets
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    model.train(train, dev)

if __name__ == "__main__":
    main()
