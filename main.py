import os
import re
import numpy as np
import urllib.request
import sklearn
from nltk.stem import PorterStemmer
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

import collections
import string

# import this for storing our BOW format
import scipy
from scipy import sparse
from sklearn import svm
from sklearn import ensemble
from sklearn.metrics import classification_report

ps = PorterStemmer()
mires = []
dict_corpora = {}


def download_file(url, file_name):
    if not os.path.exists('./' + file_name):
        print('Downloading file from {}...'.format(url))
        with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
            data = response.read()
            print('Saving file as {}...'.format(file_name))
            out_file.write(data)


def create_directory(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


def to_lower_case(input_tokens):
    return [token.lower() for token in input_tokens]


def remove_stopwords(input_tokens):
    return [token for token in input_tokens if token not in stop_words]


def apply_stemming(input_tokens):
    return [ps.stem(token) for token in input_tokens]


def tokenize(input_text):
    # split text by whitespace into words by replacing first any non-letter / non-digit character with a space
    # \w matches any [a-zA-Z0-9_] characters
    # \s matches any whitespace characters
    return re.sub(r"[^\w\s]|_", " ", input_text).split()


def preprocess(input_text):
    initial_words = tokenize(input_text)
    # convert text to lowercase
    tokens = to_lower_case(initial_words)
    # remove the stop words
    text_without_sw = remove_stopwords(tokens)
    # apply stemming
    stemmed_words = apply_stemming(text_without_sw)

    return stemmed_words


def compute_mutual_info(N, N00, N01, N11, N10):
    a = N11 + N01
    aa = N10 + N11
    b = N00 + N01
    bb = N10 + N00

    part1 = np.log2(N * N11 / (aa * a)) if N * N11 != 0 and aa * a != 0 else 0
    part2 = np.log2(N * N01 / (b * a)) if N * N01 != 0 and a * b != 0 else 0
    part3 = np.log2(N * N10 / (aa * bb)) if N * N10 != 0 and aa * bb != 0 else 0
    part4 = np.log2(N * N00 / (b * bb)) if N * N00 != 0 and b * bb != 0 else 0

    return N11 / N * part1 + N01 / N * part2 + N10 / N * part3 + N00 / N * part4


def compute_chi_squared(N, N00, N01, N11, N10):
    a = N11 + N01
    aa = N10 + N11
    b = N00 + N01
    bb = N10 + N00

    if a * aa * b * bb == 0:
        return 0

    return N * np.square(N11 * N00 - N10 * N01) / (a * aa * b * bb)


def precision_score(retrieved, relevant):
    """Precision at cutoff 10
    Args:
        retrieved (dict): The retrieved documents
        relevant (dict): The relevant documents
    @return precision (float): The precision value
    """
    tp = list(set(retrieved).intersection(relevant))
    precision = len(tp) / len(retrieved)

    return float("{0:.3f}".format(precision))


def recall_score(retrieved, relevant):
    """Recall at cutoff 50
    Args:
        retrieved (dict): The retrieved documents
        relevant (dict): The relevant documents
    @return recall (float): The recall value
    """
    tp = list(set(retrieved).intersection(relevant))
    recall = len(tp) / len(relevant)

    return float("{0:.3f}".format(recall))


def avg_precision_score(total_docs, relevant):
    """Average precision
    Args:
        retrieved (dict): The retrieved documents
        relevant (dict): The relevant documents
    @returns AP (float): The average precision value
    """
    ap = 0

    for res in range(1, len(total_docs)):
        if total_docs[res - 1][0] in relevant:
            k_retrieved = [x[0] for x in total_docs[:res]]
            x = precision_score(k_retrieved, relevant)
            ap += x * 1

    ap = ap / len(relevant)

    return float("{0:.3f}".format(ap))


def nDCG(retrieved, relevant):
    """Normalized discount cumulative gain at cutoff k
    Args:
        retrieved (dict): The retrieved documents
        relevant (dict): The relevant documents
    @return nDCG (float): The nDCG value
    """
    first_doc = retrieved[0][0]
    rel1 = int(dict(relevant)[first_doc]) if first_doc in dict(relevant) else 0
    DCG = rel1

    for index in range(1, len(retrieved)):
        rank = retrieved[index][1]
        doc_id = retrieved[index][0]
        if doc_id in dict(relevant):
            DCG += int(dict(relevant)[doc_id]) / np.log2(int(rank))

    iDCG = int(relevant[0][1])

    for idx, (doc, rel) in enumerate(relevant[1:len(retrieved)]):
        iDCG += int(rel) / np.log2(int(idx) + 2)

    nDCG = DCG / iDCG

    return float("{0:.3f}".format(nDCG))


def write_to_file(elems_to_print):
    with open('./eval_results/ir_eval.csv', 'w') as f:
        for elem in elems_to_print:
            f.write(elem[0][0])
            for index in range(1, len(elem)):
                if isinstance(elem[index], str):
                    f.write(',' + elem[index])
                else:
                    f.write(',' + format(elem[index], ".3f"))
            f.write('\n')


def WorldLevel(allWords):
    orders = [['Quran', 'OT', 'NT'], ['OT', 'Quran', 'NT'], ['NT', 'Quran', 'OT']]
    chires = []

    for order in orders:
        targetlen = len(dict_corpora[order[0]])
        otherlen = len(dict_corpora[order[1]]) + len(dict_corpora[order[2]])
        N = targetlen + otherlen
        onemires = []
        onechires = []

        for term in allWords:
            N11 = 0
            for verses in dict_corpora[order[0]]:
                if term in verses:
                    N11 += 1

            if N11 != 0:
                N01 = targetlen - N11

                N10 = 0
                for corpora in order[1:]:
                    for item in dict_corpora[corpora]:
                        if term in item:
                            N10 += 1
                N00 = otherlen - N10

                mi = compute_mutual_info(N, N00, N01, N11, N10)
                chi = compute_chi_squared(N, N00, N01, N11, N10)

                onemires.append([term, mi])
                onechires.append([term, chi])

        mires.append(sorted(onemires, key=lambda x: x[-1], reverse=-True))
        chires.append(sorted(onechires, key=lambda x: x[-1], reverse=-True))

    print("MI")
    for each in mires:
        print(each[:10])

    print("CHI")
    for each in chires:
        print(each[:10])


def TopicLevel():
    all_texts = dict_corpora['OT'] + dict_corpora['NT'] + dict_corpora['Quran']

    dictionary = Dictionary(all_texts)

    corpus = [dictionary.doc2bow(text) for text in all_texts]
    lda = LdaModel(corpus, id2word=dictionary, num_topics=20, random_state=1918, passes=20)

    # LDA for OT
    dictionary_ot = Dictionary(dict_corpora['OT'])

    corpus_ot = [dictionary_ot.doc2bow(text) for text in dict_corpora['OT']]
    topics_ot = lda.get_document_topics(corpus_ot)
    topic_dic_ot = {}
    for doc in topics_ot:
        for topic in doc:
            if topic[0] not in topic_dic_ot.keys():
                topic_dic_ot[topic[0]] = topic[1]
            else:
                topic_dic_ot[topic[0]] += topic[1]

    # LDA for NT
    dictionary_nt = Dictionary(dict_corpora['NT'])

    corpus_nt = [dictionary_nt.doc2bow(text) for text in dict_corpora['NT']]
    topics_nt = lda.get_document_topics(corpus_nt)
    topic_dic_nt = {}

    for doc in topics_nt:
        for topic in doc:
            if topic[0] not in topic_dic_nt.keys():
                topic_dic_nt[topic[0]] = topic[1]
            else:
                topic_dic_nt[topic[0]] += topic[1]

    # LDA for Quran
    dictionary_quran = Dictionary(dict_corpora['Quran'])

    corpus_quran = [dictionary_quran.doc2bow(text) for text in dict_corpora['Quran']]
    topics_quran = lda.get_document_topics(corpus_quran)
    topic_dic_quran = {}

    for doc in topics_quran:
        for topic in doc:
            if topic[0] not in topic_dic_quran.keys():
                topic_dic_quran[topic[0]] = topic[1]
            else:
                topic_dic_quran[topic[0]] += topic[1]

    for k, v in topic_dic_quran.items():
        topic_dic_quran[k] = v / len(dict_corpora['Quran'])
    for k, v in topic_dic_ot.items():
        topic_dic_ot[k] = v / len(dict_corpora['OT'])
    for k, v in topic_dic_nt.items():
        topic_dic_nt[k] = v / len(dict_corpora['NT'])

    return lda, topic_dic_quran, topic_dic_nt, topic_dic_ot


def preprocess_data(data):
    chars_to_remove = re.compile(f'[{string.punctuation}]')

    documents = []
    categories = []
    vocab = set([])

    lines = data.split('\n')

    for line in lines:
        # make a dictionary for each document
        # word_id -> count (could also be tf-idf score, etc.)
        line = line.strip()
        if line:
            # split on tabs, we have 3 columns in this tsv format file
            category, verses = line.split('\t')

            words = chars_to_remove.sub('', verses).lower().split()
            for word in words:
                vocab.add(word)

            documents.append(words)
            categories.append(category)

    return documents, categories, vocab


def convert_to_bow_matrix(preprocessed_data, word2id):
    # matrix size is number of docs x vocab size + 1 (for OOV)
    matrix_size = (len(preprocessed_data), len(word2id) + 1)
    oov_index = len(word2id)
    # matrix indexed by [doc_id, token_id]
    X = scipy.sparse.dok_matrix(matrix_size)

    # iterate through all documents in the dataset
    for doc_id, doc in enumerate(preprocessed_data):
        for word in doc:
            # default is 0, so just add to the count for this word in this doc
            # if the word is oov, increment the oov_index
            X[doc_id, word2id.get(word, oov_index)] += 1

    return X


def compute_accuracy(predictions, true_values):
    num_correct = 0
    num_total = len(predictions)
    for predicted, true in zip(predictions, true_values):
        if predicted == true:
            num_correct += 1
    return num_correct / num_total


if __name__ == '__main__':
    DATA_DIR = 'data/'
    STOP_WORDS_FILE = 'stop_words.txt'
    TRAIN_AND_DEV = DATA_DIR + 'train_and_dev.tsv'
    TEST = DATA_DIR + 'test.tsv'
    QRELS = DATA_DIR + 'qrels.csv'
    SYSTEM_RESULTS = DATA_DIR + 'system_results.csv'

    download_file(
        'http://members.unine.ch/jacques.savoy/clef/englishST.txt', STOP_WORDS_FILE)

    RESULTS_DIR = 'results/'

    # store the list of english stop words
    with open(STOP_WORDS_FILE) as f:
        stop_words = [word.strip() for word in f]

    # part 1 - IR evaluation

    docs_retrieved = {}
    docs_relevant = {}
    all_systems = set([])

    with open(QRELS) as f:
        for line in f.readlines():
            query_id, doc_id, relevance = line.split(',')
            if query_id not in docs_relevant:
                docs_relevant[query_id] = []
            details = tuple([doc_id, relevance])
            docs_relevant[query_id].append(details)

    dict_systems = {}

    with open(SYSTEM_RESULTS) as f:
        for line in f.readlines():
            system_number, query_number, doc_number, rank_of_doc, score = line.split(',')

            if system_number not in dict_systems:
                dict_systems[system_number] = dict()

            if query_number not in dict_systems[system_number]:
                dict_systems[system_number][query_number] = []

            dict_systems[system_number][query_number].append([doc_number, rank_of_doc, score])
            all_systems.add(system_number)

    ir_eval_scores = []
    systems_avg_scores = []
    res = []

    all_systems = sorted(all_systems)

    for system in all_systems:
        docs_retrieved = dict_systems.get(system)
        ir_eval_scores = []
        for query in dict_systems[system].keys():
            first_10 = [docs_retrieved[query][i][0] for i in range(10)]
            first_20 = [docs_retrieved[query][i][0] for i in range(20)]
            first_50 = [docs_retrieved[query][i][0] for i in range(50)]

            relevant = [docs_relevant[query][i][0] for i in range(len(docs_relevant[query]))]

            rprec_retrieved = [docs_retrieved[query][i][0] for i in range(len(relevant))]

            precision_10 = precision_score(first_10, relevant)
            recall_50 = recall_score(first_50, relevant)
            rprecision = precision_score(rprec_retrieved, relevant)

            ap = avg_precision_score(docs_retrieved[query], relevant)

            ndcg_10 = nDCG(docs_retrieved[query][:10], docs_relevant[query])
            ndcg_20 = nDCG(docs_retrieved[query][:20], docs_relevant[query])

            ir_eval_scores.append([precision_10, recall_50, rprecision, ap, ndcg_10, ndcg_20])
            res.append([system, query, precision_10, recall_50, rprecision, ap, ndcg_10, ndcg_20])

        systems_avg_scores = np.mean(np.array(ir_eval_scores), axis=0)
        res.append([system, 'mean', systems_avg_scores[0], systems_avg_scores[1], systems_avg_scores[2],
                    systems_avg_scores[3], systems_avg_scores[4], systems_avg_scores[5]])

    write_to_file(res)

    # part 2 - text analysis

    vocab = []
    allWords = []

    with open(TRAIN_AND_DEV) as f:
        for line in f.readlines():
            corpora, verses = line.strip().split('\t')

            if corpora not in dict_corpora:
                dict_corpora.setdefault(corpora, []).append(verses)
            else:
                dict_corpora[corpora].append(verses)

        for key in dict_corpora:
            for index, verses in enumerate(dict_corpora[key]):
                words = preprocess(verses)
                dict_corpora[key][index] = words
                vocab.extend(words)

    allWords = set(vocab)
    # WorldLevel(allWords)

    # run LDA model on three corpora
    # lda, topic_dic_Quran, topic_dic_NT, topic_dic_OT = TopicLevel()

    # rank the topics for each corpus
    # topic_ranked_NT = sorted(topic_dic_NT.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:20]
    #
    # topic_ranked_OT = sorted(topic_dic_OT.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:20]
    #
    # topic_ranked_Quran = sorted(topic_dic_Quran.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:20]
    #
    # print("ordered topics for NT")
    # for topic in topic_ranked_NT:
    #     print("topic_id: " + str(topic[0]) + ", score: " + str(topic[1]))
    #     print(lda.print_topic(topic[0]))
    #
    # print("ordered topics for OT")
    # for topic in topic_ranked_OT:
    #     print("topic_id: " + str(topic[0]) + ", score: " + str(topic[1]))
    #     print(lda.print_topic(topic[0]))
    #
    # print("ordered topics for quran")
    # for topic in topic_ranked_Quran:
    #     print("topic_id: " + str(topic[0]) + ", score: " + str(topic[1]))
    #     print(lda.print_topic(topic[0]))

    # part 3 - text classification

    training_data = open(TRAIN_AND_DEV, encoding="latin-1").read()
    test_data = open(TEST, encoding="latin-1").read()

    preprocessed_training_data, training_categories, train_vocab = preprocess_data(training_data)
    preprocessed_test_data, test_categories, test_vocab = preprocess_data(test_data)

    print(f"Training Data has {len(preprocessed_training_data)} " +
          f"documents and vocab size of {len(train_vocab)}")
    print(f"Test Data has {len(preprocessed_test_data)} " +
          f"documents and vocab size of {len(test_vocab)}")
    print(f"There were {len(set(training_categories))} " +
          f"categories in the training data and {len(set(test_categories))} in the test.")

    print(collections.Counter(training_categories).most_common())

    # convert the vocab to a word id lookup dictionary
    # anything not in this will be considered "out of vocabulary" OOV

    word2id = {}
    for word_id, word in enumerate(train_vocab):
        word2id[word] = word_id

    cat2id = {}
    for cat_id, cat in enumerate(set(training_categories)):
        cat2id[cat] = cat_id

    X_train = convert_to_bow_matrix(preprocessed_training_data, word2id)
    y_train = [cat2id[cat] for cat in training_categories]

    model = sklearn.svm.LinearSVC(C=1000)
    model.fit(X_train, y_train)

    # make a prediction
    sample_text = ['christ', 'israel', 'jesus', 'god', 'a', 'resurrection']
    # create just a single vector as input (as a 1 x V matrix)
    sample_x_in = scipy.sparse.dok_matrix((1, len(word2id) + 1))
    for word in sample_text:
        sample_x_in[0, word2id[word]] += 1

    # what does the example document look like?
    print(sample_x_in)
    prediction = model.predict(sample_x_in)
    # what category was predicted?
    print("Prediction was:", prediction[0])
    # what category was that?
    print(cat2id)

    y_train_predictions = model.predict(X_train)
    accuracy = compute_accuracy(y_train_predictions, y_train)
    print("Accuracy:", accuracy)

    X_test = convert_to_bow_matrix(preprocessed_test_data, word2id)
    y_test = [cat2id[cat] for cat in test_categories]

    y_test_predictions = model.predict(X_test)
    accuracy = compute_accuracy(y_test_predictions, y_test)
    print("Accuracy:", accuracy)

    cat_names = []
    for cat, cid in sorted(cat2id.items(), key=lambda x: x[1]):
        cat_names.append(cat)
    print(classification_report(y_test, y_test_predictions, target_names=cat_names))

    baseline_predictions = [cat2id['NT']] * len(y_test)
    baseline_accuracy = compute_accuracy(baseline_predictions, y_train)
    print("Accuracy:", baseline_accuracy)

    model = sklearn.ensemble.RandomForestClassifier()
    model.fit(X_train, y_train)

    y_train_predictions = model.predict(X_train)
    print("Train accuracy was:", compute_accuracy(y_train_predictions, y_train))
    y_test_predictions = model.predict(X_test)
    print("Test accuracy was:", compute_accuracy(y_test_predictions, y_test))

    cat_names = []
    for cat, cid in sorted(cat2id.items(), key=lambda x: x[1]):
        cat_names.append(cat)
    print(classification_report(y_test, y_test_predictions, target_names=cat_names))