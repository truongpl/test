*************
Structure of source code:
*************

├── core
│   └── __init__.py
├── engine
│   ├── baseClass.py
│   ├── __init__.py
│   └── ner.py
├── main.py
├── ocr
│   └── __init__.py
├── README.md
├── scripts
│   ├── buildDict.py
│   ├── __init__.py
│   ├── load.py
│   └── train.py
└── utilities
    ├── commonUtils.py
    ├── config.py
    ├── dataUtils.py
    └── __init__.py


A. Core:
	Provide api to interact with high layer

B. Engine:
	Core of NLP system, each module must base on baseClass: ner,sentimemt.py

C. Ocr:
	OCR module, use for image processing tasks

D. Scripts:
	Executing script to train/load NLP model

E Utilities:
	Support module


*************
Usage
*************

0. Install packages in requirements.txt via pip
1. Copy the pretrained word embeddings model to data
2. Modify the config.py in Utilities to adapt with data path
3. Execute buildDict.py to build dictionary
4. Execute train.py to train a NER model
5. Execute load.py to test the trained model

 