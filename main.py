import csv
import os
import re
from math import log

import numpy as np
import urllib.request
from nltk.stem import PorterStemmer

ps = PorterStemmer()
mires = []


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
    # convert text to lowercase
    tokens = to_lower_case(input_text)
    # remove the stop words
    text_without_sw = remove_stopwords(tokens)
    # apply stemming
    stemmed_words = apply_stemming(text_without_sw)

    return stemmed_words


def compute_mutual_info(n00, n01, n11, n10):
    a = n11 + n01
    aa = n10 + n11
    b = n00 + n01
    bb = n10 + n00
    n = n00 + n01 + n10 + n11

    return (n11 / n * log(2, (n*n11)/(aa*a)) +
            n01 / n * log(2, (n*n01)/(b*a)) +
            n10 / n * log(2, (n*n10)/(aa*bb)) +
            n00 / n * log(2, (n*n00)/(b*bb)))


def compute_chi_squared(n00, n01, n11, n10):
    n = n00 + n01 + n10 + n11
    return (n * pow(2, (n11 * n00 - n10 * n01))) / ((n11+n01) * (n11+n10) * (n10 + n00) * (n01 + n00))


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


def WorldLevel(vocab):
    orders = [['Quran', 'OT', 'NT'], ['OT', 'Quran', 'NT'], ['NT', 'Quran', 'OT']]
    chires = []

    for order in orders:
        target = order[0]
        targetlen = len(dict_corpora[target])
        otherlen = len(dict_corpora[order[1]]) + len(dict_corpora[order[2]])
        N = targetlen + otherlen
        onemires = []
        onechires = []

        for term in vocab:
            N11 = 0
            for item in dict_corpora[target]:
                if term in item:
                    N11 += 1
            N01 = targetlen - N11

            N10 = 0
            for corpora in order[1:]:
                for item in dict_corpora[corpora]:
                    if term in item:
                        N10 += 1
            N00 = otherlen - N10

            N1x = N11 + N10
            Nx1 = N11 + N01
            N0x = N00 + N01
            Nx0 = N00 + N10

            sub1 = np.log2(N * N11 / (N1x * Nx1)) if N * N11 != 0 and N1x * Nx1 != 0 else 0
            sub2 = np.log2(N * N01 / (N0x * Nx1)) if N * N01 != 0 and N0x * Nx1 != 0 else 0
            sub3 = np.log2(N * N10 / (N1x * Nx0)) if N * N10 != 0 and N1x * Nx0 != 0 else 0
            sub4 = np.log2(N * N00 / (N0x * Nx0)) if N * N00 != 0 and N0x * Nx0 != 0 else 0
            mi = (N11 / N) * sub1 + (N01 / N) * sub2 + (N10 / N) * sub3 + (N00 / N) * sub4

            below = Nx1 * N1x * Nx0 * N0x
            chi = N * np.square(N11 * N00 - N10 * N01) / below if below != 0 else 0

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


if __name__ == '__main__':
    DATA_DIR = 'data/'
    STOP_WORDS_FILE = 'stop_words.txt'
    TRAIN_AND_DEV = DATA_DIR + 'train_and_dev.tsv'
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

    dict_corpora = {}
    vocab = set([])
    categories = set([])

    with open(TRAIN_AND_DEV) as f:
        tsv_file = csv.reader(f, delimiter='\t')
        for line in tsv_file:
            if line[0] not in dict_corpora:
                dict_corpora[line[0]] = set([])
            words = preprocess(line[1:])
            for word in words:
                dict_corpora[line[0]].add(word)
                vocab.add(word)

    WorldLevel(vocab)


