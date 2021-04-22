import numpy as np
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re
from bs4 import BeautifulSoup as bs
from scipy import stats
from google.colab import drive

drive.mount('/content/drive')

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def xml_to_list(filename):
    r = []
    cnt = 0
    with open('/content/drive/My Drive/Training/' + filename + '.xml','r',encoding='utf-8') as file:
        content = file.read()
        soup = bs(content)
        for items in soup.findAll("row"):
            if len(items.get("body")) != 0:
              r.append(items.get("body"))
              cnt = cnt + 1
              if cnt == 1200:
                  break
            
    return r

def word_count(words_list):
    counts = dict()
    

    for word in words_list:
        if len(word) > 12 or len(word) == 1 or word == 'im' or word == 'ive' or word == 'hasnt' or word == 'nt':
            continue
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts

#print(word_count(text))

def preprocessing(data):
    wc = []
    for i in range(len(data)):
        #print(i)
        text = str(data[i])
        if(len(text) > 0):
            text = text.lower()
            text = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", text)
            text = re.sub(r'[^\x00-\x7F]','',text)
            text = re.sub(r"(?s)<.*?>", " ", text)
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text, flags=re.MULTILINE)
            
            text = re.sub(r"&nbsp;", " ", text)
            text = re.sub(r'[-+]?\d+',' ' , text)
            text = re.sub(r"(?s)<.*?>", " ", text)
            text = re.sub(r"  "," ",text)
            text = re.sub(r"    "," ",text)
            #text = re.sub(r')
            #c = re.compile('\n')
            #text = re.sub(c,'',text)
            '''
            txt = re.sub(r'[^\x00-\x7F]','',txt)
            t = re.compile('<.*?>')
            txt = re.sub(t,'',txt)
            c = re.compile('\n')
            txt = re.sub(c,'',txt)
            c1 = re.compile('/')
            txt = re.sub(c1,' ',txt)
            c11 = re.compile('[+-]')
            txt = re.sub(c11,'',txt)
            c2 = re.compile("\d+")
            txt = re.sub(c2,' ',txt)
            '''
            #txt = txt.lower()
            txt = text.translate((str.maketrans(' ',' ',string.punctuation)))
            tt = word_tokenize(txt)
            stop_words = set(stopwords.words('english'))
            text = [word for word in tt if not word in stop_words]
            lemmatizer=WordNetLemmatizer()
            text = [lemmatizer.lemmatize(word) for word in text]
            stemmer= PorterStemmer()
            text = [stemmer.stem(word) for word in text]
            #print(len(text))
            p = word_count(text)
            #print(p)
            wc.append(p)
        else:
            p = {}
            wc.append(p)
    
    #print(wc)
    return wc
    
topics = ['Coffee','Arduino','Chess','Biology','Astronomy','Space','Anime','Cooking','Law','Windows_Phone','Wood_Working']
g = []
rr = []
for i in range(len(topics)):
    r = xml_to_list(topics[i])
    #print(len(r))
    rr.append(r)
    df = preprocessing(r[:700])
    #print(len(df))
    g.append(df)
print(len(g))

def merge_words(d):
  #d is a list of dictionaries
  all_set = set()
  for i in range(len(d)):
    kk = set()
    for j in range(len(d[i])):
      kk = kk | set(d[i][j].keys())
    all_set = all_set | kk
  return all_set

a = merge_words(g)
#print(a)
a = list(a)
#print(a[3791])
a_len = len(a)

train = []
validate = []
test = []
for i in range(len(g)):
    df_train = preprocessing(rr[i][:500])
    train.append(df_train)
    df_validate = preprocessing(rr[i][500:700])
    validate.append(df_validate)
    df_test = preprocessing(rr[i][700:])
    test.append(df_test)

def word_mat(data, word_list):
    dict_w = np.array(data)
    d_w = dict_w.reshape(dict_w.shape[0]*dict_w.shape[1])
    r = len(d_w)
    c = len(word_list)
    t_data = np.zeros((r, c))
    r = t_data.shape[0]
    c = t_data.shape[1]
    for i in range(r):
        for j in range(c):
            t_data[i][j] = d_w[i].get(word_list[j], 0)

    return d_w, t_data

d_w,train_data = word_mat(train, a)
train_data_eu = (train_data > 0).astype(int)

d_w_v, validate_data = word_mat(validate, a)
validate_data_eu = (validate_data > 0).astype(int)

def create_labels(num_of_topics):
    train = []
    validate = []
    test = []
    for i in range(num_of_topics):
        train = train + [i]*500
        validate = validate + [i]*200
        test = test + [i]*10
    return train, validate, test
Y_train, Y_validate, Y_test = create_labels(11)

def hamming_distance(a, b):
    dist = np.abs(a - b)
    d = np.sum(dist)
    return d

def euclidean_distance(a, b):
    d = np.linalg.norm(a - b)
    return d


def construct_tf_idf_vec(data, d1, row, alpha = 0, beta = 1):
  #tf calculation
  Ndw = np.sum(data, axis = 1)
  tf = np.transpose(data)/Ndw
  tf = np.transpose(tf)

  #idf calculation
  D = row
  C = np.sum(d1, axis = 0)
  idf = np.log10((D + alpha)/(C + beta) )
  idf = np.array([idf]*len(data))
  
  #tf_idf
  tf_idf = np.multiply(tf, idf)

  return tf_idf

ro = len(train_data)
tf_idf_train = construct_tf_idf_vec(train_data, train_data_eu,ro)
tf_idf_validate = construct_tf_idf_vec(validate_data,train_data_eu,ro)

def cosine_distance(a, b):
    lob = np.dot(a, b)
    hor = np.linalg.norm(a)*np.linalg.norm(b)
    return lob/hor

def KNN(X_train, Y_train, X_test, Y_test, n_neighbours, algo):
    p = []
    uniqueOutputLabels = set(Y_train)
    uniqueOutputCount = len(uniqueOutputLabels)
    
    for i in range(len(X_test)):
        allDistances = []
        for j in range(len(X_train)):
            if algo == "cosine":
                d = 1 - cosine_distance(X_test[i], X_train[j])
            elif algo == "hamming":
                d = hamming_distance(X_test[i], X_train[j])
            elif algo == "euclidean":
                d = euclidean_distance(X_test[i], X_train[j])
            allDistances.append(d)
        #allDistances.sort(key=lambda x: x[1])
        dist = np.array(allDistances)
        voteCount = np.zeros(uniqueOutputCount)
        #print(uniqueOutputCount)
        #neighbours = []
        for i in range(n_neighbours):
            idx = np.argmin(dist)
            #neighbours.append(allDistances[i][0])
            class_label = int(Y_train[idx])
            #print(class_label)
            voteCount[class_label] += 1
            dist[idx] = 999999999
        
        predictedOutput = np.argmax(voteCount)
        #allTestNeighbours.append(neighbours)
        p.append(predictedOutput)
        
        
    return p


def accuracy(pred, act):
    cnt = 0
    for i in range(len(pred)):
        if pred[i] == act[i]:
            cnt = cnt + 1
    acc = cnt/len(pred)
    return acc

e1 = KNN(train_data_eu, Y_train, validate_data_eu, Y_validate, 1, "hamming")
print(accuracy(e1, Y_validate))
e1 = KNN(train_data_eu, Y_train, validate_data_eu, Y_validate, 3, "hamming")
print(accuracy(e1, Y_validate))
e1 = KNN(train_data_eu, Y_train, validate_data_eu, Y_validate, 5, "hamming")
print(accuracy(e1, Y_validate))

e1 = KNN(train_data, Y_train, validate_data, Y_validate, 1, "euclidean")
print(accuracy(e1, Y_validate))
e1 = KNN(train_data, Y_train, validate_data, Y_validate, 3, "euclidean")
print(accuracy(e1, Y_validate))
e1 = KNN(train_data, Y_train, validate_data, Y_validate, 5, "euclidean")
print(accuracy(e1, Y_validate))

e1 = KNN(tf_idf_train, Y_train, tf_idf_validate, Y_validate, 1, "cosine")
print(accuracy(e1, Y_validate))
e1 = KNN(tf_idf_train, Y_train, tf_idf_validate, Y_validate, 3, "cosine")
print(accuracy(e1, Y_validate))
e1 = KNN(tf_idf_train, Y_train, tf_idf_validate, Y_validate, 5, "cosine")
print(accuracy(e1, Y_validate))



def tot_word_list(train_data, k):
    t = []
    for i in range(k):
        kk = train_data[i*500:(i+1)*500]
        tsum = np.sum(kk, axis = 0)
        t.append(tsum)
    return np.array(t)

tot_word = tot_word_list(train_data, 11)

def word_prob(total_word, idx, topic, smoothing_factor, V):
    Ncw1 = total_word[topic][idx]
    Nc0 = np.sum(total_word[topic])
    prob = (Ncw1 + smoothing_factor)/(Nc0 + (smoothing_factor*V))
    return prob
#print(word_prob(tot_word, idx, 0,0,0))


def naive_bayes_prob(test_dict, word_list, topic, tot_word_list,smoothing_factor, V):
    test_doc = list(test_dict.keys())
    #print(len(test_doc))
    mul = 0
    for i in range(len(test_doc)):
        idx = word_list.index(test_doc[i])
        p = word_prob(tot_word_list, idx, topic, smoothing_factor, V)
        mul = mul + np.log2(p)
    return mul
#for i in range(11):
#    print(naive_bayes_prob(d_validate[123],a,i,tot_word, 0.04, V))
a_train = merge_words(train)
V = len(a_train) 

def prediction_NB(test_dict, word_list, topic_num, tot_word_list,smoothing_factor, V):
    prob = np.zeros((topic_num))
    for i in range(topic_num):
        prob[i] = naive_bayes_prob(test_dict, word_list, i, tot_word_list,smoothing_factor, V)
    pred = np.argmax(prob)
    return pred
#prediction_NB(d_validate[444],a,9,tot_word, 0.04, V)
  

cnt = 0
for i in range(len(d_w_v)):
    p = prediction_NB(d_w_v[i],a, 11, tot_word, 0.25, V)
    if p == Y_validate[i]:
        cnt = cnt + 1
print('NB accuracy : ', (cnt/len(Y_validate)))

gg = []
for i in range(len(rr)):
    tdf = preprocessing(rr[i][:500] + rr[i][700:])
    gg.append(tdf)
    print(len(tdf))

a_tst = merge_words(gg)
a_tst = list(a_tst)
print(len(a_tst))

test_dataset = []
for i in range(50):
    tt = []
    for j in range(11):
        tt = tt + test[j][i*10 : (i+1)*10]
    test_dataset.append(np.array(tt))
    
d_w_tr, train_data = word_mat(train, a_tst)    
train_data_eu = (train_data > 0).astype(int)

tot_word = tot_word_list(train_data, 11)

acc_NB = []
for i in range(len(test_dataset)):
    cnt = 0
    td = test_dataset[i]
    for j in range(len(td)):
        p = prediction_NB(td[j], a_tst, 11, tot_word, 0.25, V)
        if p == Y_test[j]:
            cnt = cnt + 1
    accu = (cnt/len(Y_test))*100
    print('NB accuracy: iteration : ', i + 1, ': ', accu)
    acc_NB.append(accu)
    
print('-----------Naive Bayes Statistics-------------')
acc_NB = np.array(acc_NB)
print('Average Accuracy : ', np.mean(acc_NB))
print('Maximum Accuracy : ', np.max(acc_NB))
print('Minimum Accuracy : ', np.min(acc_NB))


acc_knn = []
ro = len(train_data)
tf_idf_train = construct_tf_idf_vec(train_data, train_data_eu,ro)
for t in range(50):
    td = test_dataset[t]
    test_data = np.zeros((110, len(a_tst)))
    r1 = test_data.shape[0]
    c1 = test_data.shape[1]
    for i in range(r1):
        for j in range(c1):
            test_data[i][j] = td[i].get(a_tst[j], 0)
    tf_idf_test = construct_tf_idf_vec(test_data,train_data_eu,ro)
    e1 = KNN(tf_idf_train, Y_train, tf_idf_test, Y_test, 5, "cosine")
    c = 0
    for i in range(len(Y_test)):
        if e1[i] == Y_test[i]:
            c = c + 1
    accuracy = (c/len(Y_test))*100
    print('KNN accuracy : iteration ',t + 1,' : ', accuracy)
    acc_knn.append(accuracy)

print('-----------KNN Statistics-------------')    
knn = np.array(acc_knn)
print('Average accuracy : ',np.mean(knn))
print('Maximum accuracy : ',np.max(knn))
print('Minimum accuracy : ',np.min(knn))

np.random.seed(50)
stats.ttest_rel(knn, acc_NB)