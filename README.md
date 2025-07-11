# UDA-WF
the closed-world dataset https://drive.google.com/drive/folders/1tNw0hwy1ug9kdoI855IXxNA5VviLtJbw?usp=drive_link

the open-world dataset https://drive.google.com/drive/folders/1KkrcnIMsH-oJybS8Osxo40IDmjTJ4-QE?usp=drive_link

## Model
The "UDA-WF code" directory contains the prototype of our model.

~~~
UDA-WF code/
│
├── loss/   
│   ├── IMLoss.py                # IMLoss = KL-divergence + Entropy of one element.
│   ├── SmoothCrossEntropy.py    # used in the source model training process to relieve the effect of "long-tail distribution".
│   └── weightedCrossEntropy.py  # calculating the cross-entropy between pseudo-label and predicted label,
│                                  and optimizing the cross-entropy by pseudo-label weights.
├── network/
│   └── DFnet                    # including DFnetBase model （feature extractor）and DFnetCls (classification layer).
│
├── utils/
│   ├── dataLoaderTotal.py       # loading closed-world dataset.
│   ├── dataLoaderopenTotal.py   # loading open-world dataset.
│   ├── datasetLoader.py         # loading pseudo-label, weight, and target domain dataset.
│   ├── evaluation.py            # calculating precision, recall, and F1-score.         
│   ├── pGenLabel.py             # generating the pseudo-label and corresponding weights.
│   └── util.py                  # calculating accuracy.
├── SFShot.py                    # Function of unsupervised domain adaptation.
├── source_train.py              # training source model. 
├── target_test_alter.py         # fine-tuning target model in closed-world.
└── evaluation.py                # fine-tuning target model in open-world.

~~~
Our model significantly improves deployment efficiency. Our UDA-WF uses the unsupervised softmatch domain adaptation method to overcome the cross-domain problem while reducing auxiliary data requirements by 95% and pre-training bootstrap time by 99% compared to the State-Of-The-Art (SOTA) methods.

