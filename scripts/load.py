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

    # Test a string
    sentence = "President Xin Jinping of China, on hist first state visit to the United States, showed off his\
            familiarity with American history and pop culture last night"

    wordList = sentence.strip().split(" ")
    pred = model.predict(wordList)
    print("Input sentence: ", sentence)
    print("Prediction result = ", pred)

if __name__ == "__main__":
    main()
