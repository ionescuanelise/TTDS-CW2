import os
import re
import string
import urllib.request

import numpy as np
import pandas as pd
import scipy
import sklearn
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from nltk.stem import PorterStemmer
from scipy import sparse
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

ps = PorterStemmer()
mi_res = []
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
        hdr = "system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20\n"
        f.write(hdr)
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
    chi_res = []

    for order in orders:
        N = len(dict_corpora[order[0]]) + len(dict_corpora[order[1]]) + len(dict_corpora[order[2]])

        temp_mi = []
        temp_chi = []

        for term in allWords:
            N11 = 0

            for verses in dict_corpora[order[0]]:
                if term in verses:
                    N11 += 1

            N01 = len(dict_corpora[order[0]]) - N11
            N10 = 0

            for corpora in order[1:]:
                for item in dict_corpora[corpora]:
                    if term in item:
                        N10 += 1

            N00 = len(dict_corpora[order[1]]) + len(dict_corpora[order[2]]) - N10

            mi = compute_mutual_info(N, N00, N01, N11, N10)
            chi = compute_chi_squared(N, N00, N01, N11, N10)

            temp_chi.append([term, chi])
            temp_mi.append([term, mi])

        mi_res.append(sorted(temp_mi, key=lambda x: x[-1], reverse=-True))
        chi_res.append(sorted(temp_chi, key=lambda x: x[-1], reverse=-True))

    print("MI")
    for ind, each in enumerate(mi_res):
        print(str(ind) + "10 top words for this category are:")
        for t in each[:10]:
            print(t[0] + ":" + str(round(t[1], 3)) + ", ")

    print("CHI")
    for ind, each in enumerate(chi_res):
        print(str(ind) + "10 top words for this category are:")
        for t in each[:10]:
            print(t[0] + ":" + str(round(t[1], 3)) + ", ")


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
    vocab = set([])

    for line in data:
        # make a dictionary for each document
        # word_id -> count (could also be tf-idf score, etc.)
        line = line.strip()
        if line:
            words = chars_to_remove.sub('', line).lower().split()
            for word in words:
                vocab.add(word)
            documents.append(words)

    return documents, vocab


def remove_stopwords(input_tokens):
    return [token for token in input_tokens if token not in stop_words]


def apply_stemming(input_tokens):
    return [ps.stem(token) for token in input_tokens]


def light_preprocess_data(data):
    chars_to_remove = re.compile(f'[{string.punctuation}]')

    documents = []
    vocab = set([])

    for line in data:
        line = line.strip()
        if line:
            words = chars_to_remove.sub('', line).split()
            for word in words:
                vocab.add(word)
            documents.append(words)

    return documents, vocab


def extreme_preprocess_data(data):
    chars_to_remove = re.compile(f'[{string.punctuation}]')

    documents = []
    vocab = set([])

    for line in data:
        # make a dictionary for each document
        # word_id -> count (could also be tf-idf score, etc.)
        line = line.strip()
        if line:
            words = apply_stemming(remove_stopwords(chars_to_remove.sub('', line).lower().split()))
            for word in words:
                vocab.add(word)
            documents.append(words)

    return documents, vocab


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
            X[doc_id, word2id.get(word, oov_index)] += (1 / len(doc))

    return X


def compute_accuracy(predictions, true_values):
    num_correct = 0
    num_total = len(predictions)
    for predicted, true in zip(predictions, true_values):
        if predicted == true:
            num_correct += 1
    return num_correct / num_total


def convert_to_bow_matrix_TFIDF(preprocessed_data, word2id):
    words_positions = {}

    for doc_id, line in enumerate(preprocessed_data):
        words_positions[doc_id] = list(zip(line, range(0, len(line))))

    inverted_index = create_inverted_index(words_positions)

    # matrix size is number of docs x vocab size + 1 (for OOV)
    matrix_size = (len(preprocessed_data), len(word2id) + 1)
    oov_index = len(word2id)
    # matrix indexed by [doc_id, token_id]
    X = scipy.sparse.dok_matrix(matrix_size)

    for doc_id, doc in enumerate(preprocessed_data):
        for word in doc:
            if doc_id in inverted_index[word]:
                # Frequency of term in this document
                tf = len(inverted_index[word][doc_id])
                # Number of documents in which the term appeared
                df = len(inverted_index[word])
                X[doc_id, word2id.get(word, oov_index)] += (1 + np.log10(tf)) * np.log10(len(preprocessed_data) / df)

    return X


def create_inverted_index(words_positions):
    inverted_index = {}
    for docId, values in words_positions.items():
        for (word, position) in values:
            if not inverted_index.get(word):
                inverted_index[word] = dict()
            if not inverted_index[word].get(docId):
                inverted_index[word][docId] = []
            inverted_index[word][docId].append(position)
    return inverted_index


def baseline_TFIDF(word2id, cat2id, train_data, dev_data, train_cat, dev_cat, c, mode):
    X_train = convert_to_bow_matrix_TFIDF(train_data, word2id)
    y_train = [cat2id[cat] for cat in train_cat]

    model = sklearn.svm.SVC(C=c)
    model.fit(X_train, y_train)

    y_train_predictions = model.predict(X_train)
    accuracy = compute_accuracy(y_train_predictions, y_train)
    print("Training accuracy " + mode, accuracy)

    cat_names = []
    for cat, cid in sorted(cat2id.items(), key=lambda x: x[1]):
        cat_names.append(cat)
    print(classification_report(y_train, y_train_predictions, target_names=cat_names, digits=3))

    X_dev = convert_to_bow_matrix_TFIDF(dev_data, word2id)
    y_dev = [cat2id[cat] for cat in dev_cat]

    y_dev_predictions = model.predict(X_dev)
    accuracy = compute_accuracy(y_dev_predictions, y_dev)
    print("Dev accuracy " + mode, accuracy)

    cat_names = []
    for cat, cid in sorted(cat2id.items(), key=lambda x: x[1]):
        cat_names.append(cat)
    print(classification_report(y_dev, y_dev_predictions, target_names=cat_names, digits=3))


def baseline(word2id, cat2id, train_data, dev_data, test_data, train_cat, dev_cat, test_cat, c, mode):
    X_train = convert_to_bow_matrix(train_data, word2id)
    y_train = [cat2id[cat] for cat in train_cat]

    model = sklearn.svm.SVC(C=c)
    model.fit(X_train, y_train)

    y_train_predictions = model.predict(X_train)
    accuracy = compute_accuracy(y_train_predictions, y_train)
    print("Training accuracy " + mode, accuracy)

    cat_names = []
    for cat, cid in sorted(cat2id.items(), key=lambda x: x[1]):
        cat_names.append(cat)
    print(classification_report(y_train, y_train_predictions, target_names=cat_names, digits=3))

    X_dev = convert_to_bow_matrix(dev_data, word2id)
    y_dev = [cat2id[cat] for cat in dev_cat]

    y_dev_predictions = model.predict(X_dev)
    accuracy = compute_accuracy(y_dev_predictions, y_dev)
    print("Dev accuracy " + mode, accuracy)

    cat_names = []
    for cat, cid in sorted(cat2id.items(), key=lambda x: x[1]):
        cat_names.append(cat)
    print(classification_report(y_dev, y_dev_predictions, target_names=cat_names, digits=3))

    print("The categories are: ")

    for id, cat in enumerate(set(train_cat)):
        print(cat, id)

    print("Now print the incorrect classified ones")

    incorrect = []
    for i in range(len(y_dev_predictions)):
        if y_dev_predictions[i] != y_dev[i]:
            incorrect.append(i)

    for i in incorrect[:3]:
        print(dev_data[i])
        print("predicted category: " + str(y_dev_predictions[i]))
        print("true category: " + str(dev_cat[i]))

    # X_test = convert_to_bow_matrix(test_data, word2id)
    # y_test = [cat2id[cat] for cat in test_cat]
    #
    # y_test_predictions = model.predict(X_test)
    # accuracy = compute_accuracy(y_test_predictions, y_test)
    # print("Test accuracy " + mode, accuracy)
    #
    # cat_names = []
    # for cat, cid in sorted(cat2id.items(), key=lambda x: x[1]):
    #     cat_names.append(cat)
    # print(classification_report(y_test, y_test_predictions, target_names=cat_names, digits=3))


def random_forrest(word2id, cat2id, train_data, dev_data, train_cat, dev_cat, mode):
    X_train = convert_to_bow_matrix(train_data, word2id)
    y_train = [cat2id[cat] for cat in train_cat]

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_train_predictions = model.predict(X_train)
    accuracy = compute_accuracy(y_train_predictions, y_train)
    print("Training accuracy " + mode, accuracy)

    cat_names = []
    for cat, cid in sorted(cat2id.items(), key=lambda x: x[1]):
        cat_names.append(cat)
    print(classification_report(y_train, y_train_predictions, target_names=cat_names, digits=3))

    X_dev = convert_to_bow_matrix(dev_data, word2id)
    y_dev = [cat2id[cat] for cat in dev_cat]

    y_dev_predictions = model.predict(X_dev)
    accuracy = compute_accuracy(y_dev_predictions, y_dev)
    print("Dev accuracy " + mode, accuracy)

    cat_names = []
    for cat, cid in sorted(cat2id.items(), key=lambda x: x[1]):
        cat_names.append(cat)
    print(classification_report(y_dev, y_dev_predictions, target_names=cat_names, digits=3))


def print_accuracy(y_pred, y_true, system, split):
    f = open('./classification_results/classification.csv', "a")
    if system == 'baseline' and split == 'train':
        header = "system,split,p-quran,r-quran,f-quran,p-ot,r-ot,f-ot,p-nt,r-nt,f-nt,p-macro,r-macro,f-macro\n"
        f.write(header)

    dfpred = pd.DataFrame(y_pred)
    dftrue = pd.DataFrame(y_true)
    labels = [0, 1, 2]
    precision = []
    recall = []
    F1 = []

    for label in labels:
        pred = dfpred[dfpred[0] == label]
        index_pred = pred.index.tolist()
        true = dftrue[dftrue[0] == label]
        index_true = dftrue.reindex(index=index_pred)

        precision.append(sum(np.array(pred) == np.array(index_true)) / len(pred))
        recall.append(sum(np.array(pred) == np.array(index_true)) / len(true))
        F1.append(2 * precision[label] * recall[label] / (precision[label] + recall[label]))

    macro_P = np.mean(precision)
    macro_R = np.mean(recall)
    macro_F1 = 2 * macro_P * macro_R / (macro_P + macro_R)

    line = system + ',' + split + ',' + "{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}," \
                                        "{:.3f},{:.3f}\n".format(precision[0][0], recall[0][0], F1[0][0],
                                                                 precision[1][0], recall[1][0], F1[1][0],
                                                                 precision[2][0], recall[2][0], F1[2][0],
                                                                 macro_P, macro_R, macro_F1)
    f.write(line)


if __name__ == '__main__':
    DATA_DIR = 'data/'
    STOP_WORDS_FILE = 'stop_words.txt'
    TRAIN_AND_DEV = DATA_DIR + 'train_and_dev.tsv'
    TEST = DATA_DIR + 'test.tsv'
    QRELS = DATA_DIR + 'qrels.csv'
    SYSTEM_RESULTS = DATA_DIR + 'system_results.csv'

    download_file(
        'http://members.unine.ch/jacques.savoy/clef/englishST.txt', STOP_WORDS_FILE)

    # store the list of english stop words
    with open(STOP_WORDS_FILE) as f:
        stop_words = [word.strip() for word in f]

    # part 1 - IR evaluation

    # docs_retrieved = {}
    # docs_relevant = {}
    # all_systems = set([])
    #
    # with open(QRELS) as f:
    #     for line in f.readlines():
    #         query_id, doc_id, relevance = line.split(',')
    #         if query_id not in docs_relevant:
    #             docs_relevant[query_id] = []
    #         details = tuple([doc_id, relevance])
    #         docs_relevant[query_id].append(details)
    #
    # dict_systems = {}
    #
    # with open(SYSTEM_RESULTS) as f:
    #     for line in f.readlines():
    #         system_number, query_number, doc_number, rank_of_doc, score = line.split(',')
    #
    #         if system_number not in dict_systems:
    #             dict_systems[system_number] = dict()
    #
    #         if query_number not in dict_systems[system_number]:
    #             dict_systems[system_number][query_number] = []
    #
    #         dict_systems[system_number][query_number].append([doc_number, rank_of_doc, score])
    #         all_systems.add(system_number)
    #
    # ir_eval_scores = []
    # systems_avg_scores = []
    # res = []
    #
    # all_systems = sorted(all_systems)
    #
    # for system in all_systems:
    #     docs_retrieved = dict_systems.get(system)
    #     ir_eval_scores = []
    #     for query in dict_systems[system].keys():
    #         first_10 = [docs_retrieved[query][i][0] for i in range(10)]
    #         first_20 = [docs_retrieved[query][i][0] for i in range(20)]
    #         first_50 = [docs_retrieved[query][i][0] for i in range(50)]
    #
    #         relevant = [docs_relevant[query][i][0] for i in range(len(docs_relevant[query]))]
    #
    #         rprec_retrieved = [docs_retrieved[query][i][0] for i in range(len(relevant))]
    #
    #         precision_10 = precision_score(first_10, relevant)
    #         recall_50 = recall_score(first_50, relevant)
    #         rprecision = precision_score(rprec_retrieved, relevant)
    #
    #         ap = avg_precision_score(docs_retrieved[query], relevant)
    #
    #         ndcg_10 = nDCG(docs_retrieved[query][:10], docs_relevant[query])
    #         ndcg_20 = nDCG(docs_retrieved[query][:20], docs_relevant[query])
    #
    #         ir_eval_scores.append([precision_10, recall_50, rprecision, ap, ndcg_10, ndcg_20])
    #         res.append([system, query, precision_10, recall_50, rprecision, ap, ndcg_10, ndcg_20])
    #
    #     systems_avg_scores = np.mean(np.array(ir_eval_scores), axis=0)
    #     res.append([system, 'mean', systems_avg_scores[0], systems_avg_scores[1], systems_avg_scores[2],
    #                 systems_avg_scores[3], systems_avg_scores[4], systems_avg_scores[5]])
    #
    # write_to_file(res)

    # part 2 - text analysis

    # vocab = []
    # allWords = []
    #
    # with open(TRAIN_AND_DEV) as f:
    #     for line in f.readlines():
    #         corpora, verses = line.strip().split('\t')
    #
    #         if corpora not in dict_corpora:
    #             dict_corpora.setdefault(corpora, []).append(verses)
    #         else:
    #             dict_corpora[corpora].append(verses)
    #
    #     for key in dict_corpora:
    #         for index, verses in enumerate(dict_corpora[key]):
    #             words = preprocess(verses)
    #             dict_corpora[key][index] = words
    #             vocab.extend(words)
    #
    # allWords = set(vocab)
    # WorldLevel(allWords)

    # run LDA model on three corpora
    # lda, topic_dic_Quran, topic_dic_NT, topic_dic_OT = TopicLevel()

    # rank the topics for each corpus
    # topic_ranked_NT = sorted(topic_dic_NT.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:20]
    # topic_ranked_OT = sorted(topic_dic_OT.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:20]
    # topic_ranked_Quran = sorted(topic_dic_Quran.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:20]

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

    all_data = open(TRAIN_AND_DEV, encoding="latin-1").read()
    all_test_data = open(TEST, encoding="latin-1").read()

    lines = all_data.split('\n')
    categories = []
    docs = []

    for line in lines:
        line = line.strip()
        if line:
            category, verses = line.split('\t')
            categories.append(category)
            docs.append(verses)

    docs, categories = sklearn.utils.shuffle(docs, categories)

    split = int(0.9 * len(docs))

    train_cat = categories[:split]
    dev_cat = categories[split:]

    train_data, train_vocab = preprocess_data(docs[:split])
    dev_data, dev_vocab = preprocess_data(docs[split:])

    lines = all_test_data.split('\n')
    test_categories = []
    test_docs = []

    for line in lines:
        line = line.strip()
        if line:
            category, verses = line.split('\t')
            test_categories.append(category)
            test_docs.append(verses)

    test_data, test_vocab = preprocess_data(test_docs)
    test_cat = test_categories

    word2id = {}
    for word_id, word in enumerate(train_vocab):
        word2id[word] = word_id

    cat2id = {}
    for cat_id, cat in enumerate(set(train_cat)):
        cat2id[cat] = cat_id

    # baseline_TFIDF(word2id, cat2id, train_data, dev_data, train_cat, dev_cat, 1000, "baseline with TFIDF")

    baseline(word2id, cat2id, train_data, dev_data, test_data, train_cat, dev_cat, test_cat, 1000, "baseline SVM")
    # baseline(word2id, cat2id, train_data, dev_data, test_data, train_cat, dev_cat, test_cat, 100, "SVM 100")
    # baseline(word2id, cat2id, train_data, dev_data, test_data, train_cat, dev_cat, test_cat, 10, "SVM 10")

    # random forrest classifier
    # random_forrest(word2id, cat2id, train_data, dev_data, test_data, train_cat, dev_cat, test_cat, "random forrest")

    # preprocess the data differently

    train_data, train_vocab = light_preprocess_data(docs[:split])
    dev_data, dev_vocab = light_preprocess_data(docs[split:])

    lines = all_test_data.split('\n')
    test_categories = []
    test_docs = []

    for line in lines:
        line = line.strip()
        if line:
            category, verses = line.split('\t')
            test_categories.append(category)
            test_docs.append(verses)

    word2id = {}
    for word_id, word in enumerate(train_vocab):
        word2id[word] = word_id

    cat2id = {}
    for cat_id, cat in enumerate(set(train_cat)):
        cat2id[cat] = cat_id

    # same baseline model with different processed data (stop words removed, stemming)
    # baseline(word2id, cat2id, train_data, dev_data, train_cat, dev_cat, 1000, "processed SVM")

    # SVM with C=20, not lower case words and normalised BOW matrix
    baseline(word2id, cat2id, train_data, dev_data, test_data, train_cat, dev_cat, test_cat, 20,
             "SVM 20 - normalized - not lower")






