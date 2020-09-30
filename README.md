# Utterance-level Dialogue Understanding

This repository contains pytorch implementations of the models from the paper [Utterance-level Dialogue Understanding: An Empirical Study](https://arxiv.org/pdf/2009.13902.pdf)

![Alt text](uldu.png?raw=true "Utterance-level Dialogue Understanding")

## Task Definition

Given the transcript of a conversation along with speaker information
of each constituent utterance, the utterance-level dialogue understanding (utterance-level dialogue understanding) task aims to identify the label of each utterance from a set of  pre-defined labels that can be either a set of emotions, dialogue acts, intents etc. The figure above illustrates one such
conversation between two people, where each utterance is labeled by the
underlying emotion and intent. Formally, given the input sequence of $N$ number of
utterances $[(u_1, p_1), (u_2,p_2),\dots, (u_N,p_N)]$, where each utterance $u_i=[u_{i,1},u_{i,2},\dots,u_{i,T}]$ consists of $T$ words $u_{i,j}$ and spoken by
party $p_i$, the task is to predict the label $e_i$ of
each utterance $u_i$. In this process, the classifier can also make use of the conversational context. There are also cases where not all the utterances in a dialogue have corresponding labels. 

Emotion           |  Intent
:-------------------------:|:-------------------------:
![](emo-ex1.png)  |  ![](intent-ex1.png)
![](emo-shift.png)  |  ![](intent-ex2.png)

## Data Format

The models are all trained in an end-to-end fashion. The utterances, labels, loss masks, and speaker-specific information are thus read directly from tab separated text files. All data files follow the common format:

Utterances: Each line contains tab separated dialogue id and the utterances of the dialogue.
```
train_1    How do you like the pizza here?    Perfect. It really hits the spot.
train_2    Do you have any trains into Cambridge today?    There are many trains, when would you like to depart?    I would like to leave from Kings Lynn after 9:30.
```

Lables: Each line contains tab separated dialogue id and the encoded emotion/intent/act/strategy labels of the dialogue.
```
train_1    0    1
train_2    2    0    2
```

Loss masks: Each line contains tab separated dialogue id and the loss mask information of the utterances of the dialogue. For contextual models, the whole sequence of utterance is passed as input, but utterances having loss mask of 0 are not considered for the calculation of loss and the calcualtion of classification metrics. This is required for tasks where we want to use the full contextual information as input, but don't want to classify a subset of utterances in the output. For example, in MultiWOZ intent classification, we pass the full dialogue sequence as input but don't classify the utterances coming from the system side.
```
train_1    1    1
train_2    1    0    1
```

Speakers: Each line contains tab separated dialogue id and the speakers of the dialogue.
```
train_1    0    1
train_2    0    1    0
```

## Models

We provide implementations for end-to-end without context classifier, bcLSTM and DialogueRNN models. For bcLSTM and DialogueRNN, we also provide training argument which lets you specify whether to use residual connections or not. Navigate to `roberta-end-to-end` or `glove-end-to-end` directories to use RoBERTa or GloVe based feature extractors for the models.


<!-- ![Alt text](bclstm.png?raw=true "bcLSTM framework.") -->
<!-- ![Alt text](dialoguernn.jpg?raw=true "DialogueRNN framework.") -->
<!-- ![Alt text](residual.png?raw=true "Models with residual connections.") -->

![Alt text](dc-block.png?raw=true "bcLSTM and DialogueRNN frameworks with residual connections.")


### Execution

#### Main Model (Dialogue Level)
To train and evaluate the without context classifier model and the bcLSTM/DialogueRNN model with full context and residual connections:

`python train.py --dataset [iemocap|dailydialog|multiwoz|persuasion] --classify [emotion|act|intent|er|ee] --cls-model [logreg|lstm|dialogrnn] --residual`

The `--cls-model logreg` corresponds to the without context classifier.

#### Main Model (Utterance Level)

`python train_utt_level.py --dataset [iemocap|dailydialog|multiwoz|persuasion] --classify [emotion|act|intent|er|ee] --cls-model [logreg|lstm|dialogrnn] --residual`

#### Speaker Level Models
`w/o inter` : Trained at dialogue level. To train and evaluate bcLSTM model in this setting i.e. only with context from the same speaker:

`python train_intra_speaker.py --dataset [iemocap|dailydialog|multiwoz|persuasion] --classify [emotion|act|intent|er|ee] --residual`

`w/o intra` : Trained at utterance level. To train and evaluate bcLSTM model in this setting i.e. only with context from the other speaker:

`python train_inter_speaker.py --dataset [iemocap|dailydialog|multiwoz|persuasion] --classify [emotion|act|intent|er|ee] --residual`

#### Shuffled Context and Shuffled Context with Order Prediction Models
Trained at dialogue level. To train and evaluate bcLSTM model with various shuffling strategies in train, val, test:

`python train_shuffled_context.py --dataset [iemocap|dailydialog|multiwoz|persuasion] --classify [emotion|act|intent|er|ee] --residual --shuffle [0|1|2]`

`--shuffle 0` : Shuffled context in train, val, test.

`--shuffle 1` : Shuffled context in train, val; original context in test.

`--shuffle 2` : Original context in train, val; shuffled context in test.

#### Context Control Models
Trained on utterance level. The script is `train_context_control.py`. You can specify training arguments to determine how to control the context.

### Note

If you are running GloVe-based end-to-end models, please run the scripts multiple times and average the test scores of those runs.

## Citation

`Utterance-level Dialogue Understanding: An Empirical Study. Deepanway Ghosal, Navonil Majumder, Rada Mihalcea, Soujanya Poria. arXiv preprint
arXiv:2009.13902 (2020).`

