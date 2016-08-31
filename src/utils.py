import os as os
import numpy as np
import json
from sklearn.metrics import f1_score
import operator

def refine_text(sequence, passage=True):
    if passage:
        seq=sequence.replace('. ', ' . ')
#         print 'seq:', seq
        if seq[-1]=='.':
            seq=seq[:-1]+' .'
        return seq
    else:
        if sequence[-1]=='?':
            return sequence[:-1]
        else:
            return sequence

def match(str1, str2):
    uni1=str1.split()
    uni2=str2.split()
    bigram1=[]
    for i in range(len(uni1)-1):
        bigram1.append(uni1[i]+'=='+uni1[i+1])
    bigram2=[]
    for i in range(len(uni2)-1):
        bigram2.append(uni2[i]+'=='+uni2[i+1])
        
    match_uni=len(set(uni1)&set(uni2))
    match_bi=len(set(bigram1)&set(bigram2))
    return 0.5*match_uni+match_bi
    
def filt_passage_by_question(passage_str, q_str):
    sents=passage_str.strip().split('.')
    sent2score={}
    for sent in sents:
        if len(sent)>0:
            sent2score[sent]=match(sent, q_str)
    opti_sent=max(sent2score.iteritems(), key=operator.itemgetter(1))[0]
#     if len(opti_sent.strip())==0:
#         print 
    return opti_sent

def  load_train():
    with open('/mounts/data/proj/wenpeng/Dataset/SQuAD/train-v1.0.json') as data_file:    
        data = json.load(data_file)

    doc_size=len(data['data'])
    qa_size=0
    para_size=0
    tasks=[]
    task=None
    for i in range(doc_size):#each doc
        para_size_i=len(data['data'][i]['paragraphs'])
        for j in range(para_size_i):#each paragraph
            question_size_j=len(data['data'][i]['paragraphs'][j]['qas'])
#             Q_size_list.append(question_size_j)
            paragraph=data['data'][i]['paragraphs'][j]['context']

            for q in range(question_size_j):
                task = {"C": "", "Q": "", "A": ""} 
                question_q=data['data'][i]['paragraphs'][j]['qas'][q]['question']
                answer_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][0]['text']
                
                best_q=refine_text(question_q, False)
                best_passage_sent=filt_passage_by_question(paragraph, best_q)
                task["C"] = refine_text(best_passage_sent, True)
                task["Q"] = best_q
                task["A"] = answer_q
                tasks.append(task.copy())
                

            qa_size+=question_size_j
        para_size+=para_size_i
    print 'Load train set', para_size, 'paragraphs,', qa_size, 'question-answer pairs'
    return tasks

def  load_dev_or_test():
    with open('/mounts/data/proj/wenpeng/Dataset/SQuAD/dev-v1.0.json') as data_file:    
        data = json.load(data_file)

    doc_size=len(data['data'])
    
    qa_size=0
    para_size=0
    tasks=[]
    task=None
    for i in range(doc_size):#each doc
        para_size_i=len(data['data'][i]['paragraphs'])
        for j in range(para_size_i):#each paragraph
            question_size_j=len(data['data'][i]['paragraphs'][j]['qas']) #how many questions for this paragraph
#             Q_size_list.append(question_size_j)
            paragraph=data['data'][i]['paragraphs'][j]['context']
            for q in range(question_size_j): # for each question
                task = {"C": "", "Q": "", "A": ""} 
                question_q=data['data'][i]['paragraphs'][j]['qas'][q]['question']
                
                answer_no=len(data['data'][i]['paragraphs'][j]['qas'][q]['answers'])
                q_ansSet=set()
                for ans in range(answer_no):
                    answer_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][ans]['text']
                    q_ansSet.add(answer_q.strip())

                best_q=refine_text(question_q, False)
                best_passage_sent=filt_passage_by_question(paragraph, best_q)
                task["C"] = refine_text(best_passage_sent, True)
                task["Q"] = best_q
                task["A"] = q_ansSet
                tasks.append(task.copy())
                
            qa_size+=question_size_j
#         print 'para_size_i:', para_size_i
        para_size+=para_size_i
#     pprint(len(data['data']))
#     print data['data'][0]['paragraphs'][0]
    print 'Load dev set', para_size, 'paragraphs,', qa_size, 'question-answer pairs'
    return tasks
def init_babi(fname):
    print "==> Loading data from %s" % fname
    tasks = []
    
    # every passage-question-answer is put into a dict, then return a dict list
    task = None
    for i, line in enumerate(open(fname)):
        id = int(line[0:line.find(' ')])
        if id == 1:
            task = {"C": "", "Q": "", "A": ""} 
            
        line = line.strip()
        line = line.replace('.', ' . ')
        line = line[line.find(' ')+1:]
        if line.find('?') == -1:  #find a context sentence
            task["C"] += line
        else: # find a question
            idx = line.find('?')
            tmp = line[idx+1:].split('\t')

            task["Q"] = line[:idx]
            task["A"] = tmp[1].strip()
            tasks.append(task.copy())

    return tasks


def get_babi_raw(id, test_id):
    babi_map = {
        "1": "qa1_single-supporting-fact",
        "2": "qa2_two-supporting-facts",
        "3": "qa3_three-supporting-facts",
        "4": "qa4_two-arg-relations",
        "5": "qa5_three-arg-relations",
        "6": "qa6_yes-no-questions",
        "7": "qa7_counting",
        "8": "qa8_lists-sets",
        "9": "qa9_simple-negation",
        "10": "qa10_indefinite-knowledge",
        "11": "qa11_basic-coreference",
        "12": "qa12_conjunction",
        "13": "qa13_compound-coreference",
        "14": "qa14_time-reasoning",
        "15": "qa15_basic-deduction",
        "16": "qa16_basic-induction",
        "17": "qa17_positional-reasoning",
        "18": "qa18_size-reasoning",
        "19": "qa19_path-finding",
        "20": "qa20_agents-motivations",
        "MCTest": "MCTest",
        "19changed": "19changed",
        "joint": "all_shuffled", 
        "sh1": "../shuffled/qa1_single-supporting-fact",
        "sh2": "../shuffled/qa2_two-supporting-facts",
        "sh3": "../shuffled/qa3_three-supporting-facts",
        "sh4": "../shuffled/qa4_two-arg-relations",
        "sh5": "../shuffled/qa5_three-arg-relations",
        "sh6": "../shuffled/qa6_yes-no-questions",
        "sh7": "../shuffled/qa7_counting",
        "sh8": "../shuffled/qa8_lists-sets",
        "sh9": "../shuffled/qa9_simple-negation",
        "sh10": "../shuffled/qa10_indefinite-knowledge",
        "sh11": "../shuffled/qa11_basic-coreference",
        "sh12": "../shuffled/qa12_conjunction",
        "sh13": "../shuffled/qa13_compound-coreference",
        "sh14": "../shuffled/qa14_time-reasoning",
        "sh15": "../shuffled/qa15_basic-deduction",
        "sh16": "../shuffled/qa16_basic-induction",
        "sh17": "../shuffled/qa17_positional-reasoning",
        "sh18": "../shuffled/qa18_size-reasoning",
        "sh19": "../shuffled/qa19_path-finding",
        "sh20": "../shuffled/qa20_agents-motivations",
    }
    if (test_id == ""):  # so usually test_id is the same as training id
        test_id = id 
    babi_name = babi_map[id]
    babi_test_name = babi_map[test_id]
#     babi_train_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en/%s_train.txt' % babi_name))
#     babi_test_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en/%s_test.txt' % babi_test_name))

    babi_train_raw = load_train()
    babi_test_raw = load_dev_or_test()
    return babi_train_raw, babi_test_raw

            
def load_glove(dim):
    word2vec = {}
    
    print "==> loading 300d word2vec"
#     with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove/glove.6B." + str(dim) + "d.txt")) as f:
    f=open('/mounts/data/proj/wenpeng/Dataset/word2vec_words_300d.txt', 'r')
    for line in f:    
        l = line.split()
        word2vec[l[0]] = map(float, l[1:])
            
    print "==> word2vec is loaded"
    
    return word2vec


def create_vector(word, word2vec, word_vector_size, silent=False):
    # if the word is missing from Glove, create some fake vector and store in glove!
    vector = np.random.uniform(-0.01,0.01,(word_vector_size,))
    word2vec[word] = vector
    if (not silent):
        print "utils.py::create_vector => %s is missing" % word
    return vector


def process_word(word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=True):
    if not word in word2vec:
        create_vector(word, word2vec, word_vector_size, silent)
    if not word in vocab: 
        next_index = len(vocab)
        vocab[word] = next_index
        ivocab[next_index] = word
    
    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]
    elif to_return == "onehot":
        raise Exception("to_return = 'onehot' is not implemented yet")


def get_norm(x):
    x = np.array(x)
    return np.sum(x * x)

def macrof1(str1, str2):
    vocab1=set(str1)
    vocab2=set(str2)
    vocab=vocab1|vocab2
    
    str1_labellist=[]
    str2_labellist=[]
    for word in vocab:
        if word in vocab1:
            str1_labellist.append(1)
        else:
            str1_labellist.append(0)
        if word in vocab2:
            str2_labellist.append(1)
        else:
            str2_labellist.append(0)

#     TP_pos=0.0
#     FP_pos=0.0
#     FN_pos=0.0
#     for word in vocab:
#         if word in vocab1 and word in vocab2:
#             TP_pos+=1
#         elif word in vocab1 and word not in vocab2:
#             FP_pos+=1
#         elif word not in vocab1 and word  in vocab2:
#             FN_pos+=1
#     recall=TP_pos/(TP_pos+FN_pos) if TP_pos+FN_pos > 0 else 0.0
#     precision=TP_pos/(TP_pos+FP_pos) if TP_pos+FP_pos > 0 else 0.0
#     
#     f1=2*recall*precision/(recall+precision) if recall+precision> 0 else 0.0

    return f1_score(str1_labellist, str2_labellist, average='binary')  

def MacroF1(idlist, set_idlist):

    max_f1=0.0
    for listt in set_idlist:    
        new_f1=macrof1(idlist, listt)
        if new_f1 > max_f1:
            max_f1=new_f1
#     print max_f1
    return max_f1

def detect_boundary(wordlist1, query_list):
#     print wordlist1, query_list
    try:
        start = wordlist1.index(query_list[0])
    except ValueError:
        start = len(wordlist1)
    end=max(len(wordlist1),start+len(query_list)-1)
    return [start, end]
        
    
    
    
    
