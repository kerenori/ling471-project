import pandas as pd
import string
import os
import sys

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import matplotlib.pyplot as plt

def computeAccuracy(predictions, gold_labels):
    predictions=predictions.tolist()
    gold_labels=gold_labels.tolist()
    # The assert statement will notify you if the condition does not hold.
    correct = 0
    mistakes = []
    assert len(predictions) == len(gold_labels)
    for i in range(len(predictions)):
        if predictions[i] == gold_labels[i]:
            correct += 1
        else:
            mistakes += [i]
    accuracy = correct / len(predictions)
    return (accuracy, mistakes)

def computePrecisionRecall(predictions, gold_labels, relevant_class):
    predictions = predictions.tolist()
    gold_labels = gold_labels.tolist()
    assert len(predictions) == len(gold_labels)
    num = 0
    denom = 0
    for i in range(len(predictions)):
        if predictions[i] == gold_labels[i] and predictions[i] == relevant_class:
            num += 1
        if predictions[i] == relevant_class:
            denom += 1
    precision = num / denom
    num = 0
    denom = 0
    for i in range(len(predictions)):
        if predictions[i] == gold_labels[i] and predictions[i] == relevant_class:
            num += 1
        if gold_labels[i] == relevant_class:
            denom += 1
    recall = num / denom
    return precision, recall

ROUND = 4
GOOD_REVIEW = 1
BAD_REVIEW = 0
ALPHA = 1


def my_naive_bayes(data, column):

    test_data = data[data.type=='test']

b
    train_data = data[data.type == 'train']

    X_train = train_data[column]
    y_train = train_data.label
    X_test = test_data[column]
    y_test = test_data.label


    tf_idf_vect = TfidfVectorizer(ngram_range=(1, 2))
    tf_idf_train = tf_idf_vect.fit_transform(X_train.values)
    tf_idf_test = tf_idf_vect.transform(X_test.values)

    clf = MultinomialNB(alpha=ALPHA)

    clf.fit(tf_idf_train, y_train)

    y_pred_train = clf.predict(tf_idf_train)
    y_pred_test = clf.predict(tf_idf_test)

    accuracy_test, mistake_test = computeAccuracy(y_pred_test, y_test)
    accuracy_train, mistake_train = computeAccuracy(y_pred_train, y_train)
    precision_pos_test, recall_pos_test = computePrecisionRecall(y_pred_test, y_test, 'pos')
    precision_neg_test, recall_neg_test = computePrecisionRecall(y_pred_test, y_test, 'neg')
    precision_pos_train, recall_pos_train = computePrecisionRecall(y_pred_train, y_train, 'pos')
    precision_neg_train, recall_neg_train = computePrecisionRecall(y_pred_train, y_train, 'neg')


    return pd.Series([precision_pos_train,recall_pos_train,precision_neg_train,recall_neg_train,\
        precision_pos_test,recall_pos_test,precision_neg_test,recall_neg_test, accuracy_test, accuracy_train],
        index=["precision_pos_train", "recall_pos_train","precision_neg_train","recall_neg_train",\
        "precision_pos_test","recall_pos_test","precision_neg_test","recall_neg_test","accuracy_test","accuracy_train"])


def main(argv):
    data = pd.read_csv('my_expanded_imdb.csv', index_col=[0])
    # print(data.head())  # <- Verify the format. Comment this back out once done.

    nb_original = my_naive_bayes(data,'review')
    nb_cleaned = my_naive_bayes(data,'cleaned_text')
    nb_lowercase = my_naive_bayes(data,'lowercased')
    nb_no_stop = my_naive_bayes(data,'no_stop')
    nb_lemmatized = my_naive_bayes(data,'lemmatized')

    nb_original.plot.bar(color=['g','b','g','b','g','b','g','b','y','y'])
    plt.title("OGText"); plt.tight_layout(); plt.savefig("OGText.png")
    nb_cleaned.plot.bar(color=['g','b','g','b','g','b','g','b','y','y'])
    plt.title("CleanedText"); plt.tight_layout(); plt.savefig("CleanedText.png")
    nb_lowercase.plot.bar(color=['g','b','g','b','g','b','g','b','y','y'])
    plt.title("LowercaseText"); plt.tight_layout(); plt.savefig("LowercaseText.png")
    nb_no_stop.plot.bar(color=['g','b','g','b','g','b','g','b','y','y'])
    plt.title("NostopText"); plt.tight_layout(); plt.savefig("NostopText.png")
    nb_lemmatized.plot.bar(color=['g','b','g','b','g','b','g','b','y','y'])
    plt.title("LemmatizedText"); plt.tight_layout(); plt.savefig("LemmatizedText.png")



if __name__ == "__main__":
    main(sys.argv)
