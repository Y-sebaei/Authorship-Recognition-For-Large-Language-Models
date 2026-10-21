# Authorship recognition for large language models

Tells human written text from LLM generated text, across two corpora that look
nothing like each other: Reddit comments and arXiv abstracts. Bachelor's
project, 2024.

The interesting part is not the accuracy. It is that one classifier has to work
on a rambling Reddit answer and on a dense astrophysics abstract, where almost
every surface feature that separates human from machine in one corpus means
something different in the other.

Final model: logistic regression over Word2Vec embeddings and TF-IDF features,
**87.6% accuracy and 0.87 F1 on 1,141 held out documents**.

---

## The data

5,705 documents, each labelled `human` or `LLM`.

| Column | Meaning |
|---|---|
| `text` | The document. Reddit comment or arXiv abstract. |
| `model` | `human` or `LLM`. The target. |
| `source` | `reddit` or `arxiv`. |
| `label` | Finer grained class, 0 to 3, kept for analysis and ignored in training. |
| `text_length` | Characters. Ranges from 194 to 13,220, mean 2,386. |
| `id` | Row identifier, ignored in training. |

Split 80/20 with a fixed seed into 4,564 training and 1,141 held out documents.
`label`, `source` and `id` are dropped before fitting, so the model never sees
which corpus a document came from and cannot learn "arXiv implies machine" as a
shortcut.

## Features

Two representations of the same text, used together.

**Word2Vec**, trained on this corpus rather than downloaded: 100 dimensions,
window of 5, minimum count of 1. A document vector is the mean of its word
vectors. Mean pooling throws away word order, which matters less here than it
would for sentiment, because what separates the two classes is vocabulary and
register rather than syntax.

**TF-IDF**, which keeps the specific words a model reaches for. Word2Vec puts
"nevertheless" and "however" in roughly the same place; TF-IDF notices that one
of them turns up far more often in machine text than in a person's.

Training embeddings on 5,705 documents is the obvious weakness. A pretrained
model would have better vectors for rare words. The trade was deliberate: these
embeddings are fitted to the domain, including the LLM phrasing of 2024, which
a general purpose model trained on older text represents less sharply.

## Choosing the model

PyCaret fits the candidates on identical folds, which makes the comparison worth
something. Mean across cross validation folds:

| | Accuracy | AUC | Recall | Precision | F1 | Kappa |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.8632** | 0.8444 | 0.8632 | 0.8743 | **0.8571** | 0.6892 |
| Naive Bayes | 0.8560 | 0.8449 | 0.8560 | 0.8707 | 0.8482 | 0.6700 |
| K Neighbors | 0.7868 | 0.8032 | 0.7868 | 0.7845 | 0.7844 | 0.5351 |
| Extra Trees | 0.7865 | 0.8074 | 0.7865 | 0.8409 | 0.7585 | 0.4821 |
| Decision Tree | 0.7442 | 0.6564 | 0.7442 | 0.8184 | 0.6985 | 0.3634 |
| Random Forest | 0.7442 | 0.8256 | 0.7442 | 0.8184 | 0.6985 | 0.3634 |
| Ada Boost | 0.7442 | 0.6564 | 0.7442 | 0.8184 | 0.6985 | 0.3634 |

Logistic regression and naive Bayes sit a point apart, and both beat every tree
based model by seven points or more. That ordering is the useful result: with
sparse high dimensional text features and a few thousand rows, the linear models
win and the trees converge on something close to a majority class rule. Three of
them land on exactly 0.7442 accuracy, which is what that looks like.

Logistic regression was then tuned over 10 folds and 10 candidate
configurations, 100 fits in total. Training used class rebalancing, so the
smaller class is not simply predicted away.

## Result on held out data

The 1,141 documents the model never saw during training or tuning:

| Accuracy | AUC | Recall | Precision | F1 | Kappa | MCC |
|---|---|---|---|---|---|---|
| 0.8764 | 0.8402 | 0.8764 | 0.8959 | 0.8697 | 0.7155 | 0.7454 |

Precision above recall means the model is more careful about calling something
machine written than it is about catching every instance, which is the right
direction if a false accusation costs more than a miss.

MCC of 0.745 is the number worth quoting over accuracy. It accounts for both
classes and both error types, and it does not flatter an imbalanced split.

## The three notebooks

Same pipeline, three dataset sizes, kept because the progression is the actual
story of the project.

| Notebook | Documents | What changed |
|---|---|---|
| `Bachelor2.ipynb` | 3,152 | First end to end run, with class rebalancing. |
| `Bachelor2_1.ipynb` | 4,583 | More data, same pipeline. |
| `Bachelor3.ipynb` | 5,705 | Final run, and the numbers reported above. |

Accuracy moved very little between them, which was worth knowing: the ceiling
here is the feature representation, not the sample size.

## Known limits

**One split, one seed.** Everything above comes from a single 80/20 split with
`session_id=123`. Repeating across seeds would put error bars on these numbers,
and I would not defend the third decimal place without them.

**Domain and authorship are tangled.** arXiv abstracts and Reddit comments
differ in length, register and structure before authorship is considered.
`source` is excluded from the features, but a model can rediscover it from the
text itself, so some of the 87.6% is probably domain separation wearing
authorship's clothes. A per corpus breakdown would settle it, and it is the
first thing I would add.

**2024 text.** The LLM half comes from models of that period. Detection that
leans on phrasing habits ages with the generation that produced it.

**Mean pooled embeddings discard word order.** Fine for register, weak for
anything structural. The obvious next step is a transformer encoder, which would
also make the comparison above a fairer fight.

## Running it

The notebooks were written in Colab and run top to bottom. The data file
`Bachtrain2.jsonl` is loaded in the third cell.

```
pip install pycaret gensim scikit-learn pandas numpy
jupyter notebook Bachelor3.ipynb
```

## Stack

Python, PyCaret, scikit-learn, Gensim Word2Vec, TF-IDF, pandas, NumPy, Jupyter.
