# UDA-WF
the closed-world dataset https://drive.google.com/drive/folders/1tNw0hwy1ug9kdoI855IXxNA5VviLtJbw?usp=drive_link

the open-world dataset https://drive.google.com/drive/folders/1KkrcnIMsH-oJybS8Osxo40IDmjTJ4-QE?usp=drive_link

## Model
The "UDA-WF code" directory contains the prototype of our model.

~~~
UDA-WF code/
│
├── loss/   
│   ├── IMLoss.py                # IMLoss = KL-divergence + Entropy of one element
│   ├── SmoothCrossEntropy.py    # used in the source model training process to relieve "long-tail distribution".
│   └── weightedCrossEntropy.py  # calculating the cross-entropy between pseudo-label and predicted label,
│                                  and optimizing the cross-entropy by pseudo-label weights.
├── network/
│   └── DFnet                    # including DFnetBase model （feature extractor）and DFnetCls (classification layer)
│
├── utils/
│   ├── dataLoaderTotal.py       # loading closed-world dataset.
│   ├── dataLoaderopenTotal.py   # loading open-world dataset.
│   ├── datasetLoader.py         # loading pseudo-label, weight, and target domain dataset.
│   ├── evaluation.py            # calculating precision, recall, and F1-score.         
│   ├── pGenLabel.py             # Function of unsupervised domain adaptation (generate the pseudo-label and corresponding weights).
│   └── util.py                  # calculating accuracy.
├── main.py       # the main code of our model
├── train.py      # model training setting
├── evaluate.py   # model evaluating setting
└── evaluation.py # the metrics of the performance of our model

~~~

