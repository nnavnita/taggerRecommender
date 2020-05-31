from flair.data import Sentence
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from random import randrange
from scipy.sparse import coo_matrix
from selenium import webdriver
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from time import time
from tqdm import tqdm
import json
import pandas as pd
import string
import torch

# choosing fasttext because for subword information
embedding = DocumentPoolEmbeddings([WordEmbeddings('en')])


def text_scraper(urls, file):
    """function for scraping the text of a webpage given url"""
    start = time()
    print('SCRAPING WEBPAGES...')
    # creating a new instance of google chrome
    driver = webdriver.Chrome('./chromedriver')
    pages = []

    for url in tqdm(urls):
        driver.get(url)
        # extracting the title, content and date
        title = driver.find_element_by_tag_name('h1').text
        content = driver.find_element_by_tag_name('body').text
        try:
            script_text = driver.find_element_by_xpath(
                "//script[contains(.,'effectiveDate')]").get_attribute('innerHTML')
            date = script_text.split('effectiveDate":"')[1].split('",')[0]
        except:
            try:
                script_text = driver.find_element_by_xpath(
                    "//script[contains(.,'dateModified')]").get_attribute('innerHTML')
                date = script_text.split('dateModified"::')[1].split('T')[0]
            except:
                date = ''
        if len(title.strip()) < 0:
            title = driver.find_element_by_tag_name('h1').innerHTML
        if len(content.strip()) < 0:
            content = driver.find_element_by_tag_name('body').innerHTML
        page = {'title': title, 'content': content, 'date': date}
        pages.append(page)

    with open(file, 'w') as f:
        json.dump(pages, f)
    print('TIME: '+str(time() - start)+'s')


def clean(doc):
    """function for preprocessing text document"""
    normalized = []
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    for word in doc.split():
        if word not in stop and word not in exclude:
            normalized.append(lemma.lemmatize(word.lower()))
    return ' '.join(normalized)


def sort_coo(coo_matrix):
    """function for sorting tf_idf in descending order"""
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """function to get the feature names and tf-idf score of top n items"""
    # user only topn items from vector
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vales, score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results


def lsa(items, file):
    """function performing latent semantic analysis on text documents to generate keywords"""
    start = time()
    print('GENERATING KEYWORDS...')
    # creating a dataframe from json list
    items = pd.DataFrame(items, columns=['title', 'content', 'date', 'url'])

    # preprocessing
    items['text_content_clean'] = items['title'] + ' ' + items['content']
    items['text_content_clean'] = items['text_content_clean'].apply(
        lambda x: clean(x))

    # creating a vector of word counts
    stop = set(stopwords.words('english'))
    cv = CountVectorizer(max_df=0.8, stop_words=stop,
                         max_features=10000, ngram_range=(1, 3))
    X = cv.fit_transform(items['text_content_clean'])

    # convert to a matrix of ints
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(X)
    feature_names = cv.get_feature_names()

    results = []
    for x in tqdm(range(len(items))):
        # fetch document for which keywords needs to be extracted
        doc = items['text_content_clean'][x]
        # generate tf-idf for the given document
        tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
        # sort the tf-idf vectors by descending order of scores
        sorted_items = sort_coo(tf_idf_vector.tocoo())
        # extract only the top n; n here is 15
        keywords = extract_topn_from_vector(feature_names, sorted_items, 15)

        result = items.iloc[x].to_dict()
        del result['text_content_clean']
        result['keywords'] = keywords
        results.append(result)

    with open(file, 'w') as f:
        json.dump(results, f)

    print('TIME: '+str(time() - start)+'s')


def get_sim_items(term, items, flag):
    """function to determine similar items using cosine similarity of the their embedded keywords"""
    term_sent = Sentence(' '.join([k for k, v in term['keywords'].items()]))
    embedding.embed(term_sent)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    key = ''
    if flag == "articles":
        key = 'related_articles'
    else:
        key = 'related_products'
    if key not in term.keys():
        term[key] = []
    for item in items:
        if item['keywords'] != {}:
            item_sent = Sentence(
                ' '.join([k for k, v in item['keywords'].items()]))
            embedding.embed(item_sent)
            cos_sim = cos(
                term_sent.embedding, item_sent.embedding).tolist()
            if cos_sim > 0.7:
                # if docs are at least 70% similar, we include them
                term[key].append([item['url'], cos_sim])
    term[key].sort(key=lambda x: x[1], reverse=True)
    if len(term[key]) > 6:
        term[key] = term[key][:6]
    # store term under key for items
    for item in items:
        if item['url'] in [k for [k, v] in term[key]]:
            cos_sim = [v for [k, v] in term[key]
                       if k == item['url']][0]
            if key in item.keys():
                item[key].append([term['url'], cos_sim])
            else:
                item[key] = [[term['url'], cos_sim]]
    if flag == 'articles':
        items.insert(0, term)
        return(items)
    else:
        return(term)


def sim(articles, products, file):
    """function to determine similar articles and products"""
    start = time()
    for i in tqdm(range(len(articles))):
        if len(articles[i]['keywords']) > 0:
            # for each item check with remaining items
            articles[i] = get_sim_items(articles[i], products, "products")
            articles[i:] = get_sim_items(
                articles[i], articles[i+1:], "articles")
    with open(file, 'w+') as f:
        json.dump(articles, f)
    print('TIME: '+str(time() - start)+'s')
