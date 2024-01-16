import sys
import pandas as pd
import matplotlib.pyplot as plt

def countTokens(text):
    token_counts = {}
    tokens = text.split(' ')
    for word in tokens:
        if not word in token_counts:
            token_counts[word] = 0
        token_counts[word] += 1
    token_counts = dict(sorted(token_counts.items(), key=lambda item: item[1], reverse=True))
    return token_counts


def largest_counts(data):

    pos_train_data = data[:12500]
    pos_test_data = data[25000:37500]
    neg_train_data = data[12500:25000]
    neg_test_data = data[37500:50000]

    # NB: str.cat() turns whole column into one text
    train_counts_pos_original = countTokens(pos_train_data["review"].str.cat())
    train_counts_pos_cleaned = countTokens(
        pos_train_data["cleaned_text"].str.cat())
    train_counts_pos_lowercased = countTokens(
        pos_train_data["lowercased"].str.cat())
    train_counts_pos_no_stop = countTokens(
        pos_train_data["no_stop"].str.cat())
    train_counts_pos_lemmatized = countTokens(
        pos_train_data["lemmatized"].str.cat())

    train_counts_neg_original = countTokens(neg_train_data["review"].str.cat())
    train_counts_neg_cleaned = countTokens(
        neg_train_data["cleaned_text"].str.cat())
    train_counts_neg_lowercased = countTokens(
        neg_train_data["lowercased"].str.cat())
    train_counts_neg_no_stop = countTokens(
        neg_train_data["no_stop"].str.cat())
    train_counts_neg_lemmatized = countTokens(
        neg_train_data["lemmatized"].str.cat())

    # Once the dicts are sorted, output the first 20 rows for each.
    # This is already done below, but changes may be needed depending on what you did to sort the dicts.
    # The [:19] "slicing" syntax expects a list. If you sorting call return a list (which is likely, as being sorted
    # is conceptualy a properly of LISTS,  NOT dicts),
    # you may want to remove the additional list(dict_name.items()) conversion.
    with open('counts.txt', 'w') as f:
        f.write('Original POS reviews:\n')
        pd.Series(train_counts_pos_original).head(20).plot.bar()
        plt.title("original POS reviews"); plt.savefig('originalPOS.png')
        for k, v in list(train_counts_pos_original.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('\nCleaned POS reviews:\n')
        pd.Series(train_counts_pos_cleaned).head(20).plot.bar()
        plt.title("cleaned POS reviews"); plt.savefig('cleanedPOS.png')
        for k, v in list(train_counts_pos_cleaned.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('\nLowercased POS reviews:\n')
        pd.Series(train_counts_pos_lowercased).head(20).plot.bar()
        plt.title("lowercase POS reviews"); plt.savefig('lowercasePOS.png')
        for k, v in list(train_counts_pos_lowercased.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('\nNo stopwords POS reviews:\n')
        pd.Series(train_counts_pos_no_stop).head(20).plot.bar()
        plt.title("no stop POS reviews"); plt.savefig('nostopPOS.png')
        for k, v in list(train_counts_pos_no_stop.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('\nLemmatized POS reviews:\n')
        pd.Series(train_counts_pos_lemmatized).head(20).plot.bar()
        plt.title("lemmatized POS reviews"); plt.savefig('lemmatizedPOS.png')
        for k, v in list(train_counts_pos_lemmatized.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))

        f.write("\n------------\n")
        f.write('\nOriginal NEG reviews:\n')
        pd.Series(train_counts_neg_original).head(20).plot.bar()
        plt.title("original NEG reviews"); plt.savefig('originalNEG.png')
        for k, v in list(train_counts_neg_original.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('\nCleaned NEG reviews:\n')
        pd.Series(train_counts_neg_cleaned).head(20).plot.bar()
        plt.title("cleaned NEG reviews"); plt.savefig('cleanedNEG.png')
        for k, v in list(train_counts_neg_cleaned.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('\nLowercased NEG reviews:\n')
        pd.Series(train_counts_neg_lowercased).head(20).plot.bar()
        plt.title("lowercase NEG reviews"); plt.savefig('lowercaseNEG.png')
        for k, v in list(train_counts_neg_lowercased.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('\nNo stopwords NEG reviews:\n')
        pd.Series(train_counts_neg_no_stop).head(20).plot.bar()
        plt.title("no stop NEG reviews"); plt.savefig('nostopNEG.png')
        for k, v in list(train_counts_neg_no_stop.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('\nLemmatized NEG reviews:\n')
        pd.Series(train_counts_neg_lemmatized).head(20).plot.bar()
        plt.title("lemmatized NEG reviews"); plt.savefig('lemmatizedNEG.png')
        for k, v in list(train_counts_neg_lemmatized.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))


def main(argv):
    data = pd.read_csv(argv[1], index_col=[0])
    # print(data.head())  # <- Verify the format. Comment this back out once done.

    largest_counts(data)


if __name__ == "__main__":
    main(sys.argv)