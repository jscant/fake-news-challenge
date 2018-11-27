# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 14:50:23 2018

@author: Jack
"""


import numpy as np
import re
import sys
import time
from matplotlib import pyplot as plt
import scipy.stats as stats
import pickle
from heapq import nlargest


stem=False

# NLTK porter stemming alorithm. Setup instructions in report. Results in
# report are for stemmed corpus, but script will still run without nltk
# installed
nltk_present=True
try:
    from nltk.stem.porter import PorterStemmer
    porter_stemmer = PorterStemmer() 
    porter_stemmer.stem('testing')
except Exception:
    print('NLTK not found. Proceeding without stemming.')
    print('WARNING: RESULTS IN REPORT WERE OBTAINED USING NLTK STEMMING')
    stem=False
    nltk_present=False
else:
    print('NLTK found.')

# STANCE CODES:
# AGREE: 0
# DISAGREE: 1
# DISCUSS: 2
# UNRELATED: 3

stop_words = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all",
             "almost", "alone", "along", "already", "also", "always", "am", "among", "amongst",
             "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway",
             "anywhere", "are", "around", "as", "at", "back", "be", "became", "because", "become", "becomes",
             "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides",               
             "between", "beyond", "bill", "both", "bottom", "by", "call", "can", "cannot", "cant", "co",
             "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due",
             "during", "each", "eg", "eight", "either", "eleven", "elsewhere", "empty", "enough", "etc",
             "even", "ever", "every", "everyone", "everything", "everywhere", "few", "fifteen",
             "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found",
             "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have",
             "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
             "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed",
             "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least",
             "less", "ltd", "made", "many", "me", "meanwhile", "mill", "mine", "more",
             "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely",
             "next", "nine", "nobody", "none", "noone", "nothing",
             "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other",
             "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per", 
             "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious",
             "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
             "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system",
             "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there",
             "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thin",
             "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to",
             "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until",
             "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
             "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
             "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose",
             "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself",
             "yourselves", "the"]

idf_dict = 0
n_largest_vec = -1
adl = 0

def plt_confusion_mtx(res_mtx, title, savefig=False):
    """
    Plots the confusion matrix using matplotlib and optionally saves as .eps
    """
    plt.figure()
    for norm in [False, True]:
        idx = 1 if norm else 2
        plt.subplot(1, 2, idx)
        s=''
        if norm:
            res_mtx = res_mtx.astype('float64') / res_mtx.sum()
            s='Normalised'
        else:
            s='Absolute Values'
        
        plt.imshow(res_mtx, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(s)
        
        plt.yticks(np.arange(4), ['Agree', 'Disagree', 'Discuss', 'Unrelated'])
        plt.xticks(np.arange(4), ['Agree', 'Disagree', 'Discuss', 'Unrelated'])#, rotation='vertical')
        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        if norm:
            frmt = '.2f'
        else:
            frmt = 'd'
        
        for i in range(4):
            for j in range(4):
                plt.text(j, i, format(res_mtx[i, j], frmt), color='white' if
                         res_mtx[i, j] > res_mtx.max()/2 else 'black',
                         horizontalalignment='center',
                         verticalalignment='center')
                
        plt.tight_layout()
        plt.xlabel('Actual stance')
        plt.ylabel('Predicted stance')
    plt.show()
    if savefig:
        plt.savefig(title.replace(' ', '-') + '.eps', format='eps', dpi=1000,
                    bbox_inches='tight')

# Dictionaries have O(1) lookup time
def list_to_dict(l):
    result = {}
    for i in range(len(l)):
        result.update({l[i] : 1})
    return result

stop_words = list_to_dict(stop_words)

def add_to_dict(d, word, count=1):
    """
    Custom function for updating dictionaries to avoid extensive try/catch blocks
    when generating bag of words dictionaries
    """
    try:
        d.update({word.lower() : d[word.lower()] + count})
    except KeyError:
        d.update({word.lower() : count})

# My trusty timer class
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        end = time.time()
        self.interval = end - self.start

# Sanity check whilst parsing data files
def is_index(s):
    s = s.replace('\n', '')
    try:
        int(s)
    except ValueError:
        return False
    if s.find('.') != -1 or s.find(' ') != -1: # or s.find('-') != -1):
        return False
    return True


def histogram(set, stances=[0], categories=11, key='cosine', savefig=False):
    """
    Plots histograms for distributions of vector space and language model
    metrics (e.g. cosine similarity, bm25 etc)
    """
    plt.figure()
    S = ['Agree', 'Disagree', 'Discuss', 'Unrelated']
    for idx, stance in enumerate(stances):
        plt.subplot(2, 2, idx+1)
        cosines = []
        for entry in set:
            if entry['stance'] == stance:
                cosines.append(np.linalg.norm(entry[key]))
        h = np.array(sorted(cosines))
        cosines = np.array(cosines)
        fit = stats.norm.pdf(h, np.mean(h), np.std(h))
        plt.plot(h,fit,'r-')

        plt.hist(cosines, categories, normed=True, alpha=0.75, color='b',
                 fc=(0.5, 0.8, 0.96, 0.5))
        plt.title(S[idx])
        if key=='cosine':
            plt.xlabel(r'$\cos \theta$')
            plt.ylabel('Frequency')
            plt.gca().set_xlim([-0.025, 0.725])
        elif key == 'moothedkldivergence':
            plt.xlabel('Smoothed KL Divergence')
            plt.ylabel('Probability')
            plt.gca().set_xlim([-0.025, 18.025])
            plt.gca().set_ylim([0, 0.32])
        elif key == 'smoothedkldivergence':
            plt.xlabel('Smoothed KL Divergence')
            plt.ylabel('Probability')
            plt.gca().set_xlim([-0.025, 18.025])
            plt.gca().set_ylim([0, 0.4])
        elif key == 'bm25':
            plt.xlabel('bm25')
            plt.ylabel('Probability')
            plt.gca().set_xlim([-0.025, 75.025])
            if idx == 3:
                plt.gca().set_ylim([0, 0.25])
            else:
                plt.gca().set_ylim([0, 0.25])
                
    plt.tight_layout()
    if savefig:
        plt.savefig(key + '.eps', format='eps', dpi=1000, bbox_inches='tight')


def find_unique_words(*Sets):
    """
    Generate dictionaries containing counts of unique words for the body,
    headline and both combined for each entry
    """
    global global_len
    global global_dict
    global_len = 0
    L = 0
    for set in Sets:
        L += len(set)
    global_count= 0 
    with Timer() as unique_timer:

        for index, set in enumerate(Sets):
            for idx, dict in enumerate(set):
                global_count += 1
                percent = 100*global_count/L
                percent_str = str(percent).split('.')[0] + '%'
                sys.stdout.write(
                    "\rFinding unique words: " + percent_str)
                sys.stdout.flush()
                unique_words_hl = {}
                word_list = get_words(dict['headline'])
                headline_len = 0
                hl_prob = {}
                all_unique= {}
                for word, count in word_list.items():
                    add_to_dict(unique_words_hl, word, count)
                    add_to_dict(all_unique, word, count)
                    headline_len += count
                    add_to_dict(global_dict, word, count)
                    global_len += count
                for word, count in word_list.items():
                    add_to_dict(hl_prob, word, count/headline_len)
                unique_words_b = {}
                word_list = get_words(dict['body'])
                body_len = 0
                for word, count in word_list.items():
                    add_to_dict(unique_words_b, word, count)
                    add_to_dict(all_unique, word, count)
                    body_len += count
                    add_to_dict(global_dict, word, count)
                    global_len += count
                b_prob = {}
                for word, count in word_list.items():
                    add_to_dict(b_prob, word, count/body_len)
                dict.update({'headlineunique': unique_words_hl})
                dict.update({'headlinelength' : headline_len})
                dict.update({'headlineprob' : hl_prob})
                dict.update({'bodyunique': unique_words_b})
                dict.update({'bodylength' : body_len})
                dict.update({'bodyprob' : b_prob})
                dict.update({'unique' : all_unique})
    print("\nDone! (" + str(unique_timer.interval)[:6] + " s)")


def check_even_distribution(Set1, Set2, threshold=0.05):
    """
    Checks for a distribution of stances in the training and validation sets
    that relfects the overall distribution in the training set to within
    the specified tolernance ('threshold')
    """
    if len(Set1) == 0 or len(Set2) == 0:
        return False
    train_agree_frac = 0.0
    train_discuss_frac = 0.0
    train_disagree_frac = 0.0
    train_unrelated_frac = 0.0
    val_agree_frac = 0.0
    val_discuss_frac = 0.0
    val_disagree_frac = 0.0
    val_unrelated_frac = 0.0
    set_1_len = len(Set1)
    set_2_len = len(Set2)
    for dict in Set1:
        stance = dict['stance']
        if stance == 0:
            train_agree_frac += 1 / set_1_len
        elif stance == 1:
            train_disagree_frac += 1 / set_1_len
        elif stance == 3:
            train_unrelated_frac += 1 / set_1_len
        elif stance == 2:
            train_discuss_frac += 1 / set_1_len
    for dict in Set2:
        stance = dict['stance']
        if stance == 0:
            val_agree_frac += 1 / set_2_len
        elif stance == 1:
            val_disagree_frac += 1 / set_2_len
        elif stance == 3:
            val_unrelated_frac += 1 / set_2_len
        elif stance == 2:
            val_discuss_frac += 1 / set_2_len
    try:
        agree = max(train_agree_frac, val_agree_frac) / min(train_agree_frac, val_agree_frac)
        disagree = max(train_disagree_frac, val_disagree_frac) / min(train_disagree_frac, val_disagree_frac)
        unrelated = max(train_unrelated_frac, val_unrelated_frac) / min(train_unrelated_frac, val_unrelated_frac)
        discuss = max(train_discuss_frac, val_discuss_frac) / min(train_discuss_frac, val_discuss_frac)
        Thresh = threshold + 1 if threshold < 1 else threshold
        if agree > Thresh or disagree > Thresh or unrelated > Thresh or discuss > Thresh:
            return False
        print('Even distribution of stances to within ', 100 * threshold, "% achieved.")
        print('Agree:    ', str(agree)[:4])
        print('Disagree: ', str(disagree)[:4])
        print('Discuss:  ', str(discuss)[:4])
        print('Unrelated:', str(unrelated)[:4])
    except ZeroDivisionError:
        print('Warning: ZeroDevisionError, restarting sampling...')
        return False
    return True


def generate_training_sets(test_headlines, train_bodies, train_stances, val_frac,
                         Tol=0.05):
    """
    Splits the training data into a training set and a validation set. Uses
    check_even_distribution to check stances are distributed evenly; reshuffles
    data until a satisfactory distribution is achieved
    """
    if val_frac > 1 or val_frac < 0:
        raise ValueError("ValFrac must be between 0 and 1")
    ValLen = int(val_frac * len(train_stances))
    train_set = []
    val_set = []
    while not check_even_distribution(train_set, val_set, Tol):
        train_set = []
        val_set = []
        indexes = np.random.choice(np.array(list(range(len(test_headlines)))),
                                   size=ValLen, replace=False)
        for index in indexes:
            val_set.append(
                {'headline': test_headlines[index],
                 'body': train_bodies[index],
                 'stance': train_stances[index]})
        for index in range(len(test_headlines)):
            if index not in indexes:
                train_set.append(
                    {'headline': test_headlines[index],
                     'body': train_bodies[index],
                     'stance': train_stances[index]})
    return train_set, val_set
    

def calculate_vectors(entry):
    """
    Calculate BoW vector representations as numpy arrays for use in regression
    and other metrics
    """
    uniques = entry['unique']
    Headline = entry['headlineunique']
    Body = entry['bodyunique']
    headline_vec = np.zeros((len(uniques), 1))
    body_vec = np.zeros((len(uniques), 1))
    index = 0
    for word, count in uniques.items():
        try:
            headline_vec[index] = Headline[word.lower()]
        except KeyError:
            pass
        try:
            body_vec[index] = Body[word.lower()]
        except KeyError:
            pass
        index += 1
    entry.update({'headlinevec' : headline_vec})
    entry.update({'bodyvec' : body_vec})
    entry.update({'headlinecount' : np.sum(headline_vec)})
    entry.update({'bodycount' : np.sum(body_vec)})
    
  
def hl_probability(word, entry):
    """
    Probability of finding a word in the headline of an entry
    """
    try:
        return entry['headlineprob'][word]
    except KeyError:
        return 0
    
    
def b_probability(word, entry):
    """
    Probability of finding a word in the body of an entry
    """
    try:
        return entry['bodyprob'][word]
    except KeyError:
        return 0
    
    
def smoothed_ql(entry, u, Update=True):
    """
    Calculates and stores Dirchlet smoothed query likelihood for entry
    """
    lam = entry['bodylength']/(entry['bodylength'] + u)    
    result = 0.0
    for word, count in entry['headlineunique'].items():
        try:
            result += np.log(lam*b_probability(word, entry) +
                             (1-lam)*global_probability[word])
        except KeyError:
            try:
                result += np.log(lam*b_probability(word, entry) +
                                 (1-lam)*global_probability[word])
            except KeyError:
                result += np.log(lam*b_probability(word, entry) +
                                 (1-lam)*1E-7)
    if Update:
        entry.update({'smoothedquerylikelyhood' : result})
    else:
        return result
    
    
def smoothed_dl(entry, u, Update=True):
    """
    Calculates and stores Dirchlet smoothed document likelihood for entry
    """
    lam = entry['headlinelength']/(entry['headlinelength'] + u)    
    result = 0.0
    for word, count in entry['bodyunique'].items():
        try:
            result += np.log(lam*hl_probability(word, entry) +
                             (1-lam)*global_probability[word])
        except KeyError:
            pass
    if Update:
        entry.update({'smootheddocumentlikelyhood' : result})
    else:
        return result

    
def smoothed_kl(entry, u, Update=True):
    """
    Calculates and stores Dirchlet smoothed KL divergence for entry
    """
    result = 0
    lam = u/(u+entry['bodylength'])
    global global_probability
    for word, count in entry['headlineunique'].items():
        try:
            result -= (hl_probability(word, entry)*np.log(lam*b_probability(word, entry) +
                                      (1-lam)*global_probability[word]))
        except KeyError:
            try:
                result -= hl_probability(word, entry)*np.log((1-lam)*global_probability[word])
            except KeyError:
                result -= hl_probability(word, entry)*np.log((1-lam)*1E-8)
    if Update:
        entry.update({'smoothedkldivergence' : result})
    else:
        return result
    
    
def bm25(entry, k1, b):
    global idf_dict
    global adl
    q = entry['headlineunique']
    d = entry['bodyunique']
    bm25 = 0.0
    for word, count in q.items():
        idf = idf_dict[word]
        try:
            contrib = idf*d[word]*(k1 + 1)
            contrib /= d[word] + k1*(1-b+(b*entry['bodylength']/adl))
            bm25 += contrib
        except KeyError:
            pass
    entry.update({ 'bm25' : bm25 })

        
def global_uniques(*args):
    """
    Iterates through all entries looking for unique words and adding them
    as well as their counts to global_dict for use in dirchlet smoothing.
    Also calculates and stores corpus word probability for use in IDF.
    """
    global global_len
    global global_probability
    global global_dict
    global global_vec
    global global_vec_pos
    global porter_stemmer
    global stem
    global_vec = np.zeros(len(global_dict.items()))
    global_vec_pos = {}
    i = 0
    total_probability = 0.0
    for word, count in global_dict.items():
        add_to_dict(global_probability, word, count/global_len)
        total_probability += count/global_len
        global_vec[i] = count
        global_vec_pos.update({word : i})
        i += 1
        
    # Close to 1 but not exact due to floating point rounding errors.
    print('Summed global probability:', total_probability)


def parse(stances_fname, body_fname, test=False):
    """
    Custom parsing function
    """
    stance_ids = []
    headlines = []
    stances = []
    bodies = []
    with open(stances_fname, encoding='utf8') as f:
        for idx, line in enumerate(f.readlines()[1:]):
            sections = line.split(',')
            for section in sections:
                if is_index(section):
                    stance_ids.append(int(section))
                    Headline = line[:line.find("," + section)]
                    headlines.append(Headline)
                    if line[line.find("," + section) + len(section) + 2:].lower().find('unrelated') != -1:
                        stances.append(3)
                    elif line[line.find("," + section) + len(section) + 2:].lower().find('discuss') != -1:
                        stances.append(2)
                    elif line[line.find("," + section) + len(section) + 2:].lower().find('disagree') != -1:
                        stances.append(1)
                    elif line[line.find("," + section) + len(section) + 2:].lower().find('agree') != -1:
                        stances.append(0)
                    break
    article = ""
    body_ids = []
    body_bodies = []
    with open(body_fname, encoding='utf8') as f:
        for LineIdx, Line in enumerate(f.readlines()[1:]):
            if is_index(Line.split(',')[0]) and len(Line.split(',')) > 1:
                numStr = Line.split(',')[0]
                if len(article) > 0:
                    body_bodies.append(article)
                body_ids.append(int(numStr))
                article = Line[len(numStr) + 1:]
            else:
                article += Line
        body_bodies.append(article)

    for index, id in enumerate(stance_ids):
        bodies.append(body_bodies[body_ids.index(id)])
    return bodies, headlines, stances
        
        

def get_words(String, OutputType='dict', lim=None):
    """
    Returns dictionary of unique words and their counts from a string.
    Can also return a list (not used in report due to inefficiency)
    """
    global porter_stemmer
    global stem
    raw_result = re.compile('\w+').findall(String)
    result = {} if OutputType=='dict' else []
    if type(lim) == int:
        result250 = {} if OutputType=='dict' else []
    for index, word in enumerate(raw_result):
        if len(word) > 1 and word and not is_index(word):
            try:
                str(word)
            except Exception:
                pass
            else:
                if OutputType == 'dict':
                    if stem:
                        add_to_dict(result, porter_stemmer.stem(word.lower()),
                                    1)
                    else:
                        add_to_dict(result, word.lower(), 1)
                    if type(lim) == int:
                        if index < lim:
                            if stem:
                                add_to_dict(result250,
                                            porter_stemmer.stem(word.lower()), 1)
                            else:
                                add_to_dict(result250, word.lower(), 1)
                else:
                    if stem:
                        result.append(porter_stemmer.stem(word.lower()))
                    else:
                        result.append(word.lower())
    if type(lim) == int:
        return result, result250
    for word, count in stop_words.items():
        try:
            del result[word]
        except KeyError:
            pass
    return result
    
def vector_cos(v1, v2):
    """
    Calculates cosine of angle between two vectors
    """
    if np.size(v1, 0) != 1:
        v1 = v1.transpose()
    if np.size(v2, 1) != 1:
        v2 = v2.transpose()
    numerator = np.dot(v1, v2)
    try:
        numerator = numerator[0][0]
    except IndexError:
        try:
            numerator = numerator[0]
        except IndexError:
            pass
    try:
        denominator = np.linalg.norm(v1, 2) * np.linalg.norm(v2, 2)
    except Exception:
        return 0
    if denominator == 0:
        return 0
    return numerator / denominator

def calculate_cosine_similarity(entry, Update=True):
    """
    Calculates cosine similarity between headline and body of an entry
    """
    Angle = vector_cos(entry['headlinevec'], entry['bodyvec'])
    if Update:
        entry.update({'cosine' : Angle})
    else:
        return Angle

    
def idf(s):
    """
    Inverse document frequency for each unique word in corpus is stored in
    idf_dict for use in bm25. Can take up to 20 minutes.
    """
    global global_dict
    global idf_dict
    idf_dict = global_dict.copy()
    N = len(s)
    total_dlen = len(global_dict.items())
    idx = 0
    for word, count in global_dict.items():
        progress = 100*idx/total_dlen
        percent_str = str(progress).split('.')[0] + '%'
        sys.stdout.write(
            "\rCalculating IDF: " + percent_str)
        sys.stdout.flush()
        appears_in = 0
        for entry in s:
            try:
                entry['unique'][word]
            except KeyError:
                pass
            else:
                appears_in += 1
        else:
            idf_dict.update({ word : np.log( (N - appears_in + 0.5)/
                                            (appears_in + 0.5) ) })
        idx += 1        
            
# Generate vector of hypotheses        
def lin_predicted(X, w):
    y_pred = np.matmul(X, w)
    return y_pred


# Calculate MSE
def lin_cost(X, y, w):
    h = lin_predicted(X, w)
    J = (0.5/len(y))*(h - y).T.dot((h-y))
    return J.flatten()[0]


# Calculate gradient wrt each weight and subtract from weights vector
def lin_w_step(X, y, w, alpha):
    pred_y = lin_cost(X, y, w)
    grad = (alpha/y.shape[0])*np.dot(X.T, pred_y - y)
    return w - grad


# Optimise weights using gradient descent
def lin_optimise_w(X, y, w, alpha, tol):
    cost1 = -1
    cost2 = -1
    nIter = 0
    with Timer() as t:
        while (abs(cost1 - cost2) > tol or cost2 == -1):
            w_old = w
            cost1 = lin_cost(X, y, w_old)
            w = lin_w_step(X, y, w, alpha)
            cost2 = lin_cost(X, y, w)
            alphastr = '%.3e' % alpha
            if cost2 > cost1:
                alpha /= 2
                w = w_old
            else:
                alpha *= 1.2
            nIter += 1
            s = ' '*(7 - len(str(nIter)))
            cfstr = '%.5e' % cost2        
            timestr = str(time.time() - t.start).split('.')[0] + ' s'
            s2 = ' '*(12 - len(cfstr))
            s3 = ' '*(10 - len(alphastr))
            s4 = ' '*(8 - len(timestr))
            sys.stdout.write("\rIteration:" + s + str(nIter) + " || Cost:" +
                             s2 + cfstr + ' || Alpha:' + s3 + alphastr +
                             ' || Time:' + s4 + timestr)
            sys.stdout.flush()
    return w


def lin_regression(train_set, val_set, featurekey, stance, alpha, tol):
    """
    Use above lin_* functions to perform linear regression. Start at random
    weights then return weights which minimise MSE on validation set
    """
    train_X = np.ones((len(train_set), 1 + np.shape(train_set[0][featurekey])[0])).astype('float64')
    train_y = np.zeros((len(train_set), 1)).astype('float64')
    val_X = np.ones((len(val_set), 1 + np.shape(val_set[0][featurekey])[0])).astype('float64')
    val_y = np.zeros((len(val_set), 1)).astype('float64')
    
    for index, entry in enumerate(train_set):
        train_X[index, 1:] = entry[featurekey]
        if entry['stance'] == stance:
            train_y[index] = 1
    for index, entry in enumerate(val_set):
        val_X[index, 1:] = entry[featurekey]
        if entry['stance'] == stance:
            val_y[index] = 1
            
    ws = []
    trials = 5
    costs = np.zeros((trials,))
    for i in range(trials):
        w_start = 0.001*np.random.rand(1 + np.shape(train_set[0][featurekey])[0],
                                       1).astype('float64')
        w_end = lin_optimise_w(train_X, train_y, w_start, alpha, tol)
        ws.append(w_end)
        costs[i] = lin_cost(val_X, val_y, w_end)
    return ws[np.argmin(costs)]

def lin_multi_classify(X, w0, w1, w2, w3):
    """
    Choose class with highest hypothesis for each entry in feature set
    """
    pred0 = lin_predicted(X, w0)
    pred1 = lin_predicted(X, w1)
    pred2 = lin_predicted(X, w2)
    pred3 = lin_predicted(X, w3)
    
    pred0 -= np.min(pred0)
    pred0 /= np.max(pred0)
    pred1 -= np.min(pred1)
    pred1 /= np.max(pred1)
    pred2 -= np.min(pred2)
    pred2 /= np.max(pred2)
    pred3 -= np.min(pred3)
    pred3 /= np.max(pred3)
    
    result = np.zeros_like(pred0)
    for idx, p in enumerate(pred0):
        if p > pred1[idx] and p > pred2[idx] and p > pred3[idx]:
            result[idx] = 0
        elif pred1[idx] > pred2[idx] and pred1[idx] > pred3[idx] and pred1[idx] > pred0[idx]:
            result[idx] = 1
        elif pred2[idx] > pred1[idx] and pred2[idx] > pred0[idx] and pred2[idx] > pred3[idx]:
            result[idx] = 2
        elif pred3[idx] > pred1[idx] and pred3[idx] > pred2[idx] and pred3[idx] > pred0[idx]:
            result[idx] = 3
    return result

def lin_multi_mean_err(train_set, featurekey, w0, w1, w2, w3):
    """
    Generate confusion matrix
    """
    X = np.zeros((len(train_set), 1+np.shape(train_set[0][featurekey])[0])).astype('float64')
    y = np.zeros((len(train_set), 1)).astype('float64')
    for index, entry in enumerate(train_set):
        X[index, 1:] = entry[featurekey]
        y[index] = entry['stance']
    y_pred = lin_multi_classify(X, w0, w1, w2, w3)
    diff = abs(y_pred - y)
    res = np.zeros_like(diff)
    for idx in range(len(res)):
        if diff[idx] != 0:
            res[idx] = 1
    ResMtx = np.zeros((4,4))
    for idx, i in enumerate(y_pred):
        row = int(i)
        col = int(y[idx])
        ResMtx[row, col] += 1
    return ResMtx

def leave_one_out_lin_reg(train_set, val_set, test_set, featurekey, alpha):
    """
    Perform one-vs-many linear regression
    """
    w0 = lin_regression(train_set, val_set, featurekey, 0, alpha, 1E-11)
    w1 = lin_regression(train_set, val_set, featurekey, 1, alpha, 1E-11)
    w2 = lin_regression(train_set, val_set, featurekey, 2, alpha, 1E-11)
    w3 = lin_regression(train_set, val_set, featurekey, 3, alpha, 1E-11)
    ResMtx = lin_multi_mean_err(test_set, featurekey, w0, w1, w2, w3)
    return ResMtx.astype('int32')

# THE FOLLOWING ARE FOR LOGARITHMIC REGRESSION:

def sigmoid(z):
    return 1.0/(1+np.exp(-z))
    
def log_predicted(x, w):
    y_pred = np.matmul(x, w)
    return sigmoid(y_pred)

def log_cost(X, y, w):
    h = sigmoid(np.matmul(X, w))
    return (1/len(y))*(-y.T.dot(np.log(h)) - np.dot((1-y).T, np.log(1-h)))

def log_w_step(X, y, w, alpha):
    predicted_y = log_predicted(X, w)
    grad = np.dot(X.T, predicted_y - y)/y.shape[0]
    return w - alpha*grad

def log_optimise_w(X, y, w, alpha, tol):
    cost1 = -1
    cost2 = -1
    nIter = 0
    with Timer() as t:
        while (abs(cost1 - cost2) > tol or cost2 == -1):
            w_old = w
            cost1 = log_cost(X, y, w_old)
            w = log_w_step(X, y, w, alpha)
            cost2 = log_cost(X, y, w)
            alphastr = '%.3e' % alpha
            if cost2 > cost1:
                alpha /= 2
                w = w_old
            else:
                alpha *= 2
            nIter += 1
            s = ' '*(7 - len(str(nIter)))
            cfstr = '%.5e' % cost2        
            timestr = str(time.time() - t.start).split('.')[0] + ' s'
            s2 = ' '*(12 - len(cfstr))
            s3 = ' '*(10 - len(alphastr))
            s4 = ' '*(8 - len(timestr))
            sys.stdout.write("\rIteration:" + s + str(nIter) + " || Cost:" +
                             s2 + cfstr + ' || Alpha:' + s3 + alphastr +
                             ' || Time:' + s4 + timestr)
            sys.stdout.flush()
    return w

def log_regression(train_set, val_set, featurekey, stance, alpha, tol):
    train_X = np.ones((len(train_set), 1 + np.shape(train_set[0][featurekey])[0])).astype('float64')
    train_y = np.zeros((len(train_set), 1)).astype('float64')
    val_X = np.ones((len(val_set), 1 + np.shape(val_set[0][featurekey])[0])).astype('float64')
    val_y = np.zeros((len(val_set), 1)).astype('float64')
    
    for index, entry in enumerate(train_set):
        train_X[index, 1:] = entry[featurekey]
        if entry['stance'] == stance:
            train_y[index] = 1
    for index, entry in enumerate(val_set):
        val_X[index, 1:] = entry[featurekey]
        if entry['stance'] == stance:
            val_y[index] = 1
            
    ws = []
    trials = 1 # trials = 15 for report but this takes about an hour.
    costs = np.zeros((trials,))
    for i in range(trials):
        w_start = 0.0001*np.random.rand(1 + np.shape(train_set[0][featurekey])[0], 1).astype('float64')
        w_end = log_optimise_w(train_X, train_y, w_start, alpha, tol)
        ws.append(w_end)
        costs[i] = log_cost(val_X, val_y, w_end)
    return ws[np.argmin(costs)]

def log_multi_classify(X, w0, w1, w2, w3):
  
    pred0 = X.dot(w0)
    pred1 = X.dot(w1)
    pred2 = X.dot(w2)
    pred3 = X.dot(w3)
    
    pred0 -= np.min(pred0)
    pred0 /= np.max(pred0)
    pred1 -= np.min(pred1)
    pred1 /= np.max(pred1)
    pred2 -= np.min(pred2)
    pred2 /= np.max(pred2)
    pred3 -= np.min(pred3)
    pred3 /= np.max(pred3)
    
    result = np.zeros_like(pred0)
    for idx, p in enumerate(pred0):
        if p > pred1[idx] and p > pred2[idx] and p > pred3[idx]:
            result[idx] = 0
        elif pred1[idx] > pred2[idx] and pred1[idx] > pred3[idx] and pred1[idx] > pred0[idx]:
            result[idx] = 1
        elif pred2[idx] > pred1[idx] and pred2[idx] > pred0[idx] and pred2[idx] > pred3[idx]:
            result[idx] = 2
        elif pred3[idx] > pred1[idx] and pred3[idx] > pred2[idx] and pred3[idx] > pred0[idx]:
            result[idx] = 3
    return result

def log_multi_mean_err(set, featurekey, w0, w1, w2, w3):
    X = np.zeros((len(set), 1+np.shape(set[0][featurekey])[0])).astype('float64')
    y = np.zeros((len(set), 1)).astype('float64')
    for index, entry in enumerate(set):
        X[index, 1:] = entry[featurekey]
        y[index] = entry['stance']
    y_pred = log_multi_classify(X, w0, w1, w2, w3)
    diff = abs(y_pred - y)
    res = np.zeros_like(diff)
    for idx in range(len(res)):
        if diff[idx] != 0:
            res[idx] = 1
    ResMtx = np.zeros((4,4))
    for idx, i in enumerate(y_pred):
        row = int(i)
        col = int(y[idx])
        ResMtx[row, col] += 1
        
    return y_pred, y, np.mean(res), ResMtx


def save_object(Obj, Name):
    with open('obj/'+ Name + '.pkl', 'wb') as f:
        pickle.dump(Obj, f, pickle.HIGHEST_PROTOCOL)

def load_object(Name):
    with open('obj/' + Name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def n_largest(N, Dict):
    return nlargest(N, Dict, key=Dict.get)

def print_mean(key):
    QL = [[], [], [], []]
    for item in train_set:
        stance = item['stance']
        QL[stance].append(item[key])
    
    for i in range(4):
        QL[i] = float(np.mean(QL[i]))
    
    print('\nMean for key:', key)
    print('---------------------')
    print('Agree:    ', QL[0])
    print('Disagree: ', QL[1])
    print('Discuss:  ', QL[2])
    print('Unrelated:', QL[3])


def leave_one_out_log_reg(train_set, val_set, test_set, featurekey, alpha):
    w0 = log_regression(train_set, val_set, featurekey, 0, alpha, 1E-6)
    print()
    w1 = log_regression(train_set, val_set, featurekey, 1, alpha, 1E-6)
    print()
    w2 = log_regression(train_set, val_set, featurekey, 2, alpha, 1E-6)
    print()
    w3 = log_regression(train_set, val_set, featurekey, 3, alpha, 1E-6)
    print()
    a, b, c, ResMtx = log_multi_mean_err(test_set, featurekey, w0, w1, w2, w3)
    return w0, w1, w2, w3, ResMtx.astype('int32')

    
def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

def sm_regression(set, featurekey, alpha, tol, lam):
    X = np.zeros((len(set), 1 + np.shape(set[0][featurekey])[0])).astype('float64')
    X[:, 0] = 1
    y = np.zeros((len(set), 4)).astype('float64')
    W = np.zeros((np.shape(set[0][featurekey])[0] + 1, 4)).astype('float64')
    W = np.zeros((np.shape(set[0][featurekey])[0] + 1, 4)).astype('float64')
    for index, entry in enumerate(set):
        X[index, 1:] = entry[featurekey]
        y[index, entry['stance']] = 1
    W = sm_optimise_w(X, y, W, alpha, tol, lam)
    
    return W

def sm_optimise_w(X, y, W, alpha, tol, lam):
    cost1 = -1
    cost2 = -1
    nIter = 0
    with Timer() as t:
        while abs(cost1 - cost2) > tol or cost2 == -1:
            W_old = W
            cost1 = sm_cost(X, y, W_old, lam)
            grad_w = sm_grad(X, y, W, alpha, lam)
            W -= alpha*grad_w
            cost2 = sm_cost(X, y, W, lam)
            alphastr = '%.3e' % alpha
            if cost2 > cost1:
                alpha /= 2
                W = W_old
            else:
                alpha *= 1.3
            nIter += 1
            s = ' '*(7 - len(str(nIter)))
            cfstr = '%.5e' % cost2        
            timestr = str(time.time() - t.start).split('.')[0] + ' s'
            s2 = ' '*(12 - len(cfstr))
            s3 = ' '*(10 - len(alphastr))
            s4 = ' '*(8 - len(timestr))
            sys.stdout.write("\rIteration:" + s + str(nIter) + " || Cost:" + s2
                             +cfstr + ' || Alpha:' + s3 + alphastr +
                             ' || Time:' + s4 + timestr)
            sys.stdout.flush()
    return W

def sm_cost(X, y, W, lam):
    m = y.shape[0]
    reg = (lam/2)*np.sum(W*W)
    y_pred = softmax(np.matmul(X, W))
    J = -(1/m)*np.sum(np.log(y_pred)*y) + reg
    return J

def sm_grad(X, y, W, alpha, lam):
    m = y.shape[0]
    y_pred = softmax(np.matmul(X, W))
    grad = -(1/m)*np.dot(X.T, (y - y_pred)) + lam*W
    return alpha*grad

def sm_results(set, featurekey, W):
    X = np.zeros((len(set), 1 + np.shape(set[0][featurekey])[0])).astype('float64')
    X[:, 0] = 1
    y = np.zeros((len(set), 1)).astype('int32')
    for index, entry in enumerate(set):
        X[index, 1:] = entry[featurekey]
        y[index] = entry['stance']
    pred = np.argmax(X.dot(W), axis=1).flatten()
    ResMtx = np.zeros((4, 4)).astype('int32')
    for i in range(len(pred)-1):
        row = pred[i]
        col = y[i]
        ResMtx[row, col] += 1
    return ResMtx


# Generate feature vector for use in regression
def feature_vec(common_words_pos, entry):
    """
    Calculates the feature vector for use with regression for headline/body
    pair.
    """
    
    result = np.zeros((4004, ))
    result[0] = entry['cosine']
    result[1] = entry['smoothedkldivergence']
    result[2] = entry['bm25']
    result[3] = entry['smoothedquerylikelyhood']
    for word, count in entry['headlineunique'].items():
        try:
            result[common_words_pos[word] + 4] = count
        except KeyError:
            pass
    for word, count in entry['bodyunique'].items():
        try:
            result[common_words_pos[word] + 2004] = count
        except KeyError:
            pass
            
    entry.update({'featurevec' : result})
    

def Initialise(train_stances_filename, train_bodies_filename, test_stances_filename, test_bodies_filename, test,
               val_frac, tol, Stemming=False):
    global stem
    global global_probability
    global global_len
    global global_dict
    global global_vec
    global global_vec_pos
    global n_largest_vec
    global idf_dict
    global adl
    
    suffix = 'NotStemmed'
    if Stemming and nltk_present:
        stem = True
        suffix = 'Stemmed'
    else:
        stem = False
    if test:
        try:
            train_set = load_object('TrainDict' + suffix)
            test_set = load_object('TestDict' + suffix)
            val_set = load_object('ValDict' + suffix)
            global_dict = load_object('GlobalDict' + suffix)
            global_probability = load_object('GlobalProbability' + suffix)
            global_len = load_object('GlobalLen' + suffix)
            global_vec = load_object('GlobalVec' + suffix)
            global_vec_pos = load_object('GlobalVecPos' + suffix)
            idf_dict = load_object('IDFDict' + suffix)
            print('Sucessfully loaded dictionaries')
        except FileNotFoundError:
            print('Problem reading dictionaries from file. Generating from',
                  'scratch...')
            test = False
        
    if not test:
        train_set = []
        val_set = []
        test_set = []
        print('Parsing training data...')        
        train_bodies, train_headlines, train_stances = parse(train_stances_filename,
                                                           train_bodies_filename)
        print('Done!')
        print('Parsing test data...')
        test_bodies, test_headlines, test_stances = parse(test_stances_filename,
                                                          test_bodies_filename)
        
        
        print('Done!')
        for idx, headline in enumerate(test_headlines):
            test_set.append({'headline': headline, 'body': test_bodies[idx],
                             'stance' : test_stances[idx]})
                
        train_set, val_set = generate_training_sets(train_headlines, train_bodies,
                                                train_stances, val_frac, tol)
        
        find_unique_words(test_set, train_set, val_set)

        print('Finding all vectors...')
        with Timer() as e:
            for item in test_set + train_set + val_set:
                calculate_vectors(item)
        print('Done! (' + str(e.interval)[:6] + ' s)')
        
        print('Finding global unique words...')
        with Timer() as a:
            global_uniques(train_set + test_set + val_set)
        print('Done! (' + str(a.interval)[:6] + ' s)')
        
        with Timer() as p:
            idf(val_set + train_set + test_set)
        print('\nDone! (' + str(p.interval)[:6] + ' s)')
        
        # Do not run this code unless you want 1.2gb of dictionary objects
        # stored locally! Used for faster debugging etc.
        if 1 == 2:
            try:
                save_object(train_set, 'TrainDict' + suffix)
                save_object(test_set, 'TestDict' + suffix)
                save_object(val_set, 'ValDict' + suffix)
                save_object(global_dict, 'GlobalDict' + suffix)
                save_object(global_probability, 'GlobalProbability' + suffix)
                save_object(global_len, 'GlobalLen' + suffix)
                save_object(global_vec, 'GlobalVec' + suffix)
                save_object(global_vec_pos, 'GlobalVecPos' + suffix)
                save_object(idf_dict, 'IDFDict' + suffix)
            except Exception:
                print('Problem saving dictionaries.')
            else:
                print('Saved dictionaries to obj/')
            
    adl = 0
    for entry in train_set:
        adl += entry['bodylength']
    adl /= len(train_set)
    
    print('Finding Cosine Similarities...')
    with Timer() as g:
        for item in test_set + train_set + val_set:
            calculate_cosine_similarity(item)
    print('Done! (' + str(g.interval)[:6] + ' s)')
    
    print('Finding Smoothed Query Likelyhoods...')
    with Timer() as d:
        for item in test_set + train_set + val_set:
            smoothed_ql(item, 1000)
    print('Done! (' + str(d.interval)[:6] + ' s)')
    
    print('Finding Smoothed KL Divergences...')
    with Timer() as f:
        for item in test_set + train_set + val_set:
            smoothed_kl(item, 1000)
    print('Done! (' + str(f.interval)[:6] + ' s)')
    
    print('Finding BM25...')    
    with Timer() as q:
        for entry in train_set + test_set + val_set:
            bm25(entry, k1=1.2, b=0.75)
    print('Done! (' + str(q.interval)[:6] + ' s)')
    
    
    print('Finding feature vectors...')    
    with Timer() as b:
        common_words_pos = {}
        common_words = n_largest(2000, global_dict)
        idx = 0
        for word in common_words:
            common_words_pos.update({word: idx})
            idx += 1
        for entry in test_set + val_set + train_set:        
            feature_vec(common_words_pos, entry)
    print('Done! (' + str(b.interval)[:6] + ' s)')
        
    return test_set, train_set, val_set

with Timer() as InitTimer:
    global_probability = {}
    global_len = 0
    global_dict = {}
    global_vec = 0
    global_vec_pos = 0
    test_set, train_set, val_set = Initialise('train_stances.csv', 'train_bodies.csv',
                                           'competition_test_stances.csv',
                                           'competition_test_bodies.csv',
                                           test=False, val_frac=0.1, tol=0.05,
                                           Stemming=True)
     
print('\n---------------------------------------------')
print('INITIALISATION COMPLETE. Total time: ' + str(InitTimer.interval)[:6] + " s")
print('---------------------------------------------\n')


# All graphing and other fun stuff from here:
histogram(train_set, stances=[0, 1, 2, 3], categories=20, key='cosine', savefig=False)
histogram(train_set, stances=[0, 1, 2, 3], categories=30, key='smoothedkldivergence', savefig=False)
histogram(train_set, stances=[0, 1, 2, 3], categories=50, key='bm25', savefig=False)

print('Performing linear regression (5 validation trials per weight vector)...')    
with Timer() as x:
    lin_reg_results = leave_one_out_lin_reg(
            train_set, val_set, test_set, 'featurevec', 0.01)
print('Done! (' + str(x.interval)[:6] + ' s)\n')
plt_confusion_mtx(lin_reg_results, 'Linear Regression', savefig=False)


print('Performing logistic regression...')# (5 validation trials per weight vector)...')    
with Timer() as x:
    w0, w1, w2, w3, log_reg_results = leave_one_out_log_reg(train_set, val_set,
                                                            test_set, 'featurevec', 0.001)
print('Done! (' + str(x.interval)[:6] + ' s)\n')
plt_confusion_mtx(log_reg_results, 'Logistic Regression', savefig=False)

# WARNING: Tolernace used in report is 1e-8, which takes 40 minutes. tol=1e-6
# performs slightly worse but only takes 5 minutes.
print('Performing softmax regression...')    
with Timer() as y:
    w = sm_regression(train_set, 'featurevec', 0.001, 1E-6, 0.00001)
    sm_mtx = sm_results(test_set, 'featurevec', w)
print('\nDone! (' + str(y.interval)[:6] + ' s)\n')
plt_confusion_mtx(sm_mtx, 'Softmax Regression 1', savefig=False)

diff_cos = np.amax(w[1, :]) - np.amin(w[0, :])
diff_kl= np.amax(w[0, :]) - np.amin(w[1, :])