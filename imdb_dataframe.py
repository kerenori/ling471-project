import sys
import re
import string
from pathlib import Path

import pandas as pd
import csv

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
import nltk
from nltk.corpus import stopwords
from nltk import stem
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


'''
Write a function which accepts a list of 4 directories:
train/pos, train/neg, test/pos, and test/neg.

The result of calling this program on the 4 directories is a new .csv file in the working directory.
'''

# Constants:
POS = 1
NEG = 0

import pandas as pd

def iterateOverFiles(pathName,trainortest,posorneg):
    p = Path(pathName)
    files = list(p.glob('**/*.txt'))
    out = []
    count = 0
    count100 = 0;
    total = len(files)
    for file in files:
        count += 1
        count100 += 1
        if count100 == 100:
            print(f"Processing file {count} out of {total}")
            count100 = 0
        text, cleaned_text, lowercased, no_stop, lemmatized = cleanFileContents(file)
        out += [(file,posorneg,trainortest,text,cleaned_text,lowercased,no_stop,lemmatized)]
    return pd.DataFrame(out, columns = ["file", "label", "type", "review", "cleaned_text", "lowercased",
                                        "no_stop", "lemmatized"])
def createDataFrame(argv):
    new_filename = "my_expanded_imdb.csv"
    data = []
    #positive train
    print("1 of 4: Train POS")
    dftrainpos = iterateOverFiles(argv[1],'train','pos')

    #positive test
    print("2 of 4: Test POS")
    dftestpos = iterateOverFiles(argv[2], 'test', 'pos')

    #negative train
    print("3 of 4: Train NEG")
    dftrainneg = iterateOverFiles(argv[3], 'train', 'neg')

    #negative test
    print("4 of 4: Test NEG")
    dftestneg = iterateOverFiles(argv[4], 'test', 'neg')

    combineddf=pd.concat([dftrainpos,dftestpos,dftrainneg,dftestneg])
    combineddf.to_csv('/Users/keren3/PycharmProjects/assignment5/my_expanded_imdb.csv')

    return dftestneg, dftrainneg, dftestpos, dftrainpos, combineddf


'''
The function below should be called on a file name.
It opens the file, reads its contents, and stores it in a variable.
Then, it removes punctuation marks, and returns the "cleaned" text.
'''


def review_to_words(review, remove_stopwords=False, lemmatize=False):
    # Getting an off-the-shelf list of English "stopwords"
    stops = stopwords.words('english')
    # Initializing an instance of the NLTK stemmer/lemmatizer class
    sno = stem.SnowballStemmer('english')
    # Removing HTML using BeautifulSoup preprocessing package
    review_text = BeautifulSoup(review, "html.parser").get_text()
    # Remove non-letters using a regular expression
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    # Tokenizing by whitespace
    words = review_text.split()
    # Recall "list comprehension" from the lecture demo and try to understand what the below loops are doing:
    if remove_stopwords:
        words = [w for w in words if not w in stops]
    if lemmatize:
        lemmas = [sno.stem(w).encode('utf8') for w in words]
        # The join() function is a built-in method of strings.
        # The below says: iterate over the "lemmas" list and create
        # a new string where each item in "lemmas" is added to this new string,
        # and the items are separated by a space.
        # The b-thing is a quirk of the SnowballStemmer package.
        return b" ".join(lemmas)
    else:
        return ' '.join(words)

def cleanFileContents(f):
    with open(f, 'r', encoding='utf-8') as f:
        text = f.read()
    cleaned_text = review_to_words(text)
    lowercased = cleaned_text.lower()
    no_stop = review_to_words(lowercased, remove_stopwords=True)
    lemmatized = review_to_words(no_stop, lemmatize=True)
    return text, cleaned_text, lowercased, no_stop, lemmatized


def main(argv):
    dftestneg, dftrainneg, dftestpos, dftrainpos, combineddf = createDataFrame(argv)
    print("")

if __name__ == "__main__":
    main(sys.argv)