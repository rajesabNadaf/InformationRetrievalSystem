import re
import sys
from collections import defaultdict
import xml.etree.ElementTree as ET


stop_words_file = "stopwords-en.txt"
with open(stop_words_file, 'r', encoding='utf-8') as file:
    stop_words = set(file.read().split())

def remove_non_ascii(word):
    return ''.join([char for char in word if ord(char) < 128])

def to_lowercase(word):
    return word.lower()

def remove_punctuation(word):
    return re.sub(r'[^\w\s]', '', word)
def preprocess_text(text):
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    # Apply each preprocessing step to the tokens
    tokens = [remove_non_ascii(token) for token in tokens]
    tokens = [to_lowercase(token) for token in tokens]
    tokens = [remove_punctuation(token) for token in tokens]
    # Filter out empty tokens
    tokens = [token for token in tokens if token]

    return tokens

def index_documents(documents):
    inverted_index = defaultdict(list)
    for doc_id, text in documents.items():
        tokens = preprocess_text(text)
        for position, term in enumerate(tokens):
            inverted_index[term].append((doc_id, position))
    return inverted_index

def read_xml_file(file_path):
    documents = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        xml_content = file.read().strip()  # Trim whitespaces
        root = ET.fromstring(xml_content)
        i =0
        for doc in root.findall('doc'):
            doc_id = doc.find('docno').text.strip()
            try:
                text = doc.find('text').text.strip()
                documents[doc_id] = text
            except Exception as e:
                i=i+1
                print("doc_id" , i , "XML Block With issue :", ET.tostring(doc, encoding="unicode"))
                continue

    return documents



xml_file_path = "cran.all.1400.xml"
documents = read_xml_file(xml_file_path)
inverted_index = index_documents(documents)




#------------VSM Model Implemention-------------------------------

def calculate_tf(term_freq):
    return 1 + math.log(term_freq, 10) if term_freq > 0 else 0

def calculate_idf(term, inverted_index, total_docs):
    doc_freq = len(inverted_index.get(term, []))
    return math.log(total_docs / (1 + doc_freq), 10) if doc_freq > 0 else 0

def calculate_tf_idf(term_freq, term_idf):
    return term_freq * term_idf

def preprocess_query(query):
    return preprocess_text(query)

def search_documents_vsm(query_id, query, inverted_index, documents, output_file):
    query_tokens = preprocess_query(query)
    doc_scores = defaultdict(float)
    idf_scores = {term: calculate_idf(term, inverted_index, len(documents)) for term in query_tokens}
    for term in query_tokens:
        query_tf_idf = calculate_tf_idf(1, idf_scores[term])
        if term in inverted_index:
            for doc_id, tf in inverted_index[term]:
                doc_scores[doc_id] += calculate_tf_idf(tf, idf_scores[term])

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    with open(output_file, 'a', encoding='utf-8') as f:
        for i, (doc_id, score) in enumerate(sorted_docs[:10000], 1):
            f.write(f"{query_id} Q0 {doc_id} {i} {score} VSM\n")

    print(f"Retrieved Documents for Query {query_id} have been written to {output_file}")


output_file = "vsm_output.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("")

def read_queries_xml(file_path):
    queries = []
    with open(file_path, 'r', encoding='utf-8') as file:
        xml_content = file.read().strip()  # Trim whitespaces
        root = ET.fromstring(xml_content)
        for top in root.findall('top'):
            num = top.find('num').text.strip()
            title = top.find('title').text.strip()
            queries.append((num, title))
    return queries

# Example usage:
queries_file_path = "cran.qry.xml"
queries = read_queries_xml(queries_file_path)
print("Queries from cran.qry.xml:")
for num, title in queries:
    print(f"Query {num}: {title}")

# Iterate through each query and search documents using VSM
for query_id, (num, query) in enumerate(queries, 1):
    search_documents_vsm(num, query, inverted_index, documents, output_file)

#---------------BM25----------------------------

from math import log


def calculate_bm25(query, doc_id, inverted_index, documents, k1=1.5, b=0.75):

    score = 0
    doc_length = len(documents[doc_id])
    avg_doc_length = sum(len(doc) for doc in documents.values()) / len(documents)

    for term in query:
        df = len(inverted_index.get(term, []))
        if df == 0:
            continue

        tf = sum(tf for doc, tf in inverted_index[term] if doc == doc_id)

        idf = log((len(documents) - df + 0.5) / (df + 0.5) + 1, 2)
        tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))

        score += idf * tf_component

    return score


def search_documents_bm25(query_id, query, inverted_index, documents, output_file):

    query_tokens = preprocess_query(query)

    doc_scores = defaultdict(float)

    for doc_id in documents:
        score = calculate_bm25(query_tokens, doc_id, inverted_index, documents)
        doc_scores[doc_id] = score

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    with open(output_file, 'a', encoding='utf-8') as f:
        for i, (doc_id, score) in enumerate(sorted_docs[:1000], 1):
            f.write(f"{query_id} Q0 {doc_id} {i} {score} BM25\n")

    print(f"Retrieved Documents for Query {query_id} have been written to {output_file}")


output_file = "bm25_output.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("")

for query_id, (num, query) in enumerate(queries, 1):
    search_documents_bm25(num, query, inverted_index, documents, output_file)


#------------------RLM Model -----------------------
import re
import math
import numpy as np
from collections import defaultdict
import xml.etree.ElementTree as ET

def calculate_tf(term, document):
    tokens = document.split()
    term_freq = tokens.count(term)
    return 1 + math.log(term_freq, 10) if term_freq > 0 else 0

def calculate_idf(term, inverted_index, total_docs):
    doc_freq = len(inverted_index.get(term, []))
    return math.log(total_docs / (1 + doc_freq), 10) if doc_freq > 0 else 0

def calculate_tf_idf(term, document, inverted_index, total_docs):
    tf = calculate_tf(term, document)
    idf = calculate_idf(term, inverted_index, total_docs)
    return tf * idf

def index_documents(documents):
    inverted_index = defaultdict(list)
    for doc_id, text in documents.items():
        tokens = preprocess_text(text)
        for position, term in enumerate(tokens):
            inverted_index[term].append((doc_id, position))
    return inverted_index

def preprocess_text(text):
    tokens = re.findall(r'\b\w+\b', text)
    tokens = [token.lower() for token in tokens if token.isalnum()]
    stop_words = set(['a', 'an', 'the', 'in', 'of', 'and', 'to', 'for', 'on', 'with', 'at', 'by', 'as', 'it'])
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

def read_xml_file(file_path):
    documents = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        xml_content = file.read().strip()
        root = ET.fromstring(xml_content)
        for doc in root.findall('doc'):
            doc_id = doc.find('docno').text.strip()
            text = doc.find('text').text.strip()
            documents[doc_id] = text
    return documents

def train_rlm_model(documents, relevance_labels, inverted_index, total_docs):
    rlm_model = {}
    for term in inverted_index.keys():
        term_scores = []
        for doc_id, text in documents.items():
            score = calculate_tf_idf(term, text, inverted_index, total_docs)
            term_scores.append(score)
        rlm_model[term] = np.array(term_scores)
    return rlm_model

def rank_documents_rlm(query, rlm_model, documents, total_docs):
    query_terms = preprocess_text(query)
    doc_scores = defaultdict(float)
    for term in query_terms:
        if term in rlm_model:
            term_scores = rlm_model[term]
            doc_scores.update({doc_id: doc_scores[doc_id] + score for doc_id, score in zip(documents.keys(), term_scores)})
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs

xml_file_path = "cran.all.1400.xml"
queries_file_path = "cran.qry.xml"

documents = read_xml_file(xml_file_path)

queries = read_queries_xml(queries_file_path)

relevance_labels = np.random.randint(0, 2, len(documents))

inverted_index = index_documents(documents)

rlm_model = train_rlm_model(documents, relevance_labels, inverted_index, len(documents))

inverted_index = index_documents(documents)

output_file_rlm = "rlm_output.txt"
with open(output_file_rlm, 'w', encoding='utf-8') as f:
    for query_id, query_text in queries:
        ranked_docs = rank_documents_rlm(query_text, rlm_model, documents, len(documents))
        for rank, (doc_id, score) in enumerate(ranked_docs[:1000], 1):
            f.write(f"{query_id} Q0 {doc_id} {rank} {score} RLM\n")


#-------- TREC EVAL ---

import pytrec_eval


with open('cranqrel.trec.txt', 'r') as f:
    qrels = pytrec_eval.parse_qrel(f)

with open('vsm_output.txt', 'r') as f:
    vsm_results = pytrec_eval.parse_run(f)

with open('bm25_output.txt', 'r') as f:
    bm25_results = pytrec_eval.parse_run(f)

with open('query_likelihood_output.txt', 'r') as f:
    rlm_results = pytrec_eval.parse_run(f)


evaluation_measures = ['map', 'P_5', 'ndcg']

evaluator = pytrec_eval.RelevanceEvaluator(qrels, evaluation_measures)

# Evaluate VSM results
vsm_eval = evaluator.evaluate(vsm_results)
print("VSM Evaluation Results:")
for measure in evaluation_measures:
    vsm_scores = [query_eval[measure] for query_eval in vsm_eval.values() if query_eval[measure] != 0]
    vsm_aggregated_measure = pytrec_eval.compute_aggregated_measure(measure, vsm_scores)
    print(f"{measure}: {vsm_aggregated_measure}")

# Evaluate BM25 results
bm25_eval = evaluator.evaluate(bm25_results)
print("\nBM25 Evaluation Results:")
for measure in evaluation_measures:
    bm25_scores = [query_eval[measure] for query_eval in bm25_eval.values() if query_eval[measure] != 0]
    bm25_aggregated_measure = pytrec_eval.compute_aggregated_measure(measure, bm25_scores)
    print(f"{measure}: {bm25_aggregated_measure}")

# Evaluate RLM results
rlm_eval = evaluator.evaluate(rlm_results)
print("\nRLM Evaluation Results:")
for measure in evaluation_measures:
    rlm_scores = [query_eval[measure] for query_eval in rlm_eval.values() if query_eval[measure] != 0]
    rlm_aggregated_measure = pytrec_eval.compute_aggregated_measure(measure, rlm_scores)
    print(f"{measure}: {rlm_aggregated_measure}")