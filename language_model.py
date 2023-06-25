import re
import numpy as np
import sys



def perplexity(prob,length):
    return np.power(1/prob, 1/length)

def wordTokenize(sentence):
    return sentence.split()
def preprocessSentence(sentence):
    corpus = sentence.lower()
    corpus = corpus.replace('\n','')
    corpus=re.sub('\#[a-zA-Z0-9]\w+','<HASHTAG>',corpus) # hashtags
    corpus = re.sub('(https?:\/\/|www\.)?\S+[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}\S+', '<URL>',corpus) #URLs
    corpus=re.sub('\@\w+','<MENTION>',corpus) # mentions
    corpus=re.sub(r'\S*[\w\~\-]\@[\w\~\-]\S*', '<EMAIL>', corpus) #emails
    corpus = re.sub(r'\d+:\d\d:?\d{0,2}?( am|am| pm|pm)', r'<TIME>', re.sub(r'\d{2,4}\-\d\d-\d{2,4}|\d{2,4}\/\d\d\/\d{2,4}|\d{2,4}:\d\d:?\d{2,4}', '<DATE>', corpus)) #Date time
    corpus = re.sub(r'(?<=\s)[\:\.]?\d*[\:\.]?\d*[\:\.]?(?=\s)', r'<NUM>', corpus)
    #I wont';t toI will and similar removals
    corpus = re.sub(r'can\'t', r'can not', corpus)
    corpus = re.sub(r'won\'t', r'will not', corpus)
    corpus = re.sub(r'([a-zA-Z]+)n\'t', r'\1 not', corpus)
    corpus = re.sub(r'([a-zA-Z]+)\'s', r'\1 is', corpus)
    corpus = re.sub(r'([iI])\'m', r'\1 am', corpus)
    corpus = re.sub(r'([a-zA-Z]+)\'ve', r'\1 have', corpus)
    corpus = re.sub(r'([a-zA-Z]+)\'d', r'\1 had', corpus)
    corpus = re.sub(r'([a-zA-Z]+)\'ll', r'\1 will', corpus)
    corpus = re.sub(r'([a-zA-Z]+)\'re', r'\1 are', corpus)
    corpus = re.sub(r'([a-zA-Z]+)in\'', r'\1ing', corpus)
    #corpus = corpus.replace("  "," ")
    corpus=re.sub('[^\w\s\<\>]+',' ',corpus) # punctuations
    return corpus


def sentenceTokenizer(lines,n):
    processedSentences=[]
    for line in lines:
        sent = preprocessSentence(line)
        if(len(sent.split())>0):
            sent = "<S> "*(n-1) + sent + " <E>"
            processedSentences.append(sent) 
    return processedSentences

def ngrams(words,n):
    ngramlist=[]
    i=0
    j=n-1
    while j<len(words):
        ngramlist.append(tuple(words[i:j+1]))
        i+=1
        j+=1
    return ngramlist


def Count_Occurences(key,d):
    if len(key)==0:
        return 0
    n = len(key)
    try:
        return d[n][key]
    except KeyError:
        return 0

def SumOfCounts(key,d):
#     count=0
#     n = len(prevSent)+1
#     for k,v in d[n].items():
#         if k[:-1]==prevSent:
#             count+=v
#     return count
    if len(key)==0:
        return 1
    n = len(key)
    try:
        return d[n][key]
    except KeyError:
        return 1


def countPositives(prevSent,d):
    count=0
    n = len(prevSent)
    try:
        if prevSent in cache[n]:
            count=len(cache[n][prevSent])
        else:
            count=0
    except:
        count=0
    if count<0:
        (a,b) = (count,False)
    else:
        (a,b)=(0.225,True)
    
    return (a,b)

def createDict(processed_sentences):
    unigrams={}
    bigrams={}
    trigrams={}
    fourgrams={}
    cache={}
    thresholdForUNK=2
    d={}
    for j in processed_sentences:
        tokens = wordTokenize(j)
        itms = ngrams(tokens,1)
        for i in itms:
            if i not in unigrams.keys():
                unigrams[i]=1
            else:
                unigrams[i]+=1
        
        itms = ngrams(tokens,2)
        cache[1]={}
        for i in itms:
            if i not in bigrams.keys():
                bigrams[i]=1
            else:
                bigrams[i]+=1
            if i[:-1] not in cache[1]:
                cache[1][i[:-1]]={tuple([i[-1]]):1}
            else:
                cache[1][i[:-1]][tuple([i[-1]])]=1
        d[2]=bigrams
        
        itms = ngrams(tokens,3)
        cache[2]={}
        for i in itms:
            if i not in trigrams.keys():
                trigrams[i]=1
            else:
                trigrams[i]+=1
            if i[:-1] not in cache[2]:
                cache[2][i[:-1]]={tuple([i[-1]]):1}
            else:
                cache[2][i[:-1]][tuple([i[-1]])]=1
        
        d[3]=trigrams
        
        itms = ngrams(tokens,4)
        cache[3]={}
        for i in itms:
            if i not in fourgrams.keys():
                fourgrams[i]=1
            else:
                fourgrams[i]+=1
            if i[:-1] not in cache[3]:
                cache[3][i[:-1]]={tuple([i[-1]]):1}
            else:
                cache[3][i[:-1]][tuple([i[-1]])]=1
        d[4]=fourgrams
    count=0
    rmKeys=[]
    for k,v in unigrams.items():
        if v<thresholdForUNK:
            count+=v
            rmKeys.append(k)
    unigrams["<UNK>"]=count
    for i in rmKeys:
        _ =unigrams.pop(i)
    d[1]=unigrams
    return d, cache


#================================== ** Kneser-ney Smoothning ** ======================================== 
def Kneser_Ney(prevSent,currWord,step,d,delta=0.75):
    n= len(prevSent)+1
    if currWord not in d[1].keys():
        #val = d[1]['<UNK>']/sumOfUnigramValues
        val = 1/len(d[1])
        return val
    
    #Empty history
    if n==1:
        # val = d[1]['<UNK>']/sumOfUnigramValues
        # return val
        val= d[1][currWord]/len(d[1])
        return val
    
    first=0
    if step==1:
        try:
            first = max(Count_Occurences(prevSent+currWord,d)-delta,0)/SumOfCounts(prevSent,d)
        except ZeroDivisionError:
            first = 0
    else:
        try:
            first = max(Count_Occurences(currWord,d)-delta,0)/len(d[n])
        except ZeroDivisionError:
            first=0
    alpha=0
    try:
        a, b = countPositives(prevSent,d)
        if b==False:
            alpha = (delta/SumOfCounts(prevSent,d))*a
        else:
            alpha = a
    except ZeroDivisionError:
        val = d[1]['<UNK>']/sumOfUnigramValues
        return val
    
    shortenedPrev = prevSent[1:]
    second = alpha*Kneser_Ney(shortenedPrev,currWord,step+1,d)
    if((first+second)== 0):
        val= d[1]['<UNK>']/sumOfUnigramValues
        return val
    
    return first+second

#================================== ** Witten Bell Smoothning ** ======================================== 
def Witten_Bell(prevSent,currWord,d):
    n = len(prevSent)+1
    if n==1:
        if currWord in d[1]:
            return Count_Occurences(currWord,d)/d[1]['<UNK>']
        val = 1/len(d[1])
        return val
    alpha=0
    try:
        a,b = countPositives(prevSent,d)
        if b==False:
            alpha = a/(a+SumOfCounts(prevSent,d))
        else:
            alpha = a
    except ZeroDivisionError:
        return 1/len(d[n])
    first = Count_Occurences(prevSent+currWord,d)/SumOfCounts(prevSent,d)
    shortenedPrev = prevSent[1:]
    ans = (1-alpha)*first + alpha*Witten_Bell(shortenedPrev,currWord,d)
    if ans == 0:
        val= d[1]['<UNK>']/sumOfUnigramValues
        return val
    return ans



def perplexity_kneser_kneys(processedSentence):
    delta = 0.75
    avgPPLX=0
    perplexityArr=[]
    for sentence in processedSentence:
        tokens = wordTokenize(sentence)
        ngramSentence = ngrams(tokens,4)
        prob=1
        arr=[]
        for i in ngramSentence:
            prev = tuple(i[:-1])
            curr = tuple([i[-1]])
            val = Kneser_Ney(prev,curr,1,d,delta)
            #print(curr,prev, val)
            #arr.append((val)**(1/(len(sentence)-4)))
            prob*=val
        # print("Probability: ",prob)
        #pplx = 1/np.prod(arr)
        pplx= perplexity(prob,len(tokens))
        perplexityArr.append(pplx)
        avgPPLX+=pplx
        #print("Perplexity: ",pplx)
    avgPPLX = avgPPLX/len(processedSentence)
    return (avgPPLX, perplexityArr)


def perplexity_Witten_Bell(processedSentence):
    avgPPLX=0
    perplexityArr=[]
    for sentence in processedSentence:
        #print(sentence)
        tokens = wordTokenize(sentence)
        ngramSentence = ngrams(tokens,4)
        prob=1
        arr = []
        for i in ngramSentence:
            prev = tuple(i[:-1])
            curr = tuple([i[-1]])
            val = Witten_Bell(prev,curr,d)
            arr.append((val)**(1/(len(tokens))))
            #prob*=val
        #print("Probability: ",prob)
        pplx = 1/np.prod(arr)
        perplexityArr.append(pplx)
        #pplx= perplexity(prob,len(sentence)-4)
        avgPPLX+=pplx
        #print("Perplexity: ",pplx)
    avgPPLX = avgPPLX/len(processedSentence)
    return (avgPPLX,perplexityArr)

def calculateProb_Witten_Bell(sentence):
    avgPPLX=0
    perplexityArr=[]
    #print(sentence)
    tokens = wordTokenize(sentence)
    ngramSentence = ngrams(tokens,4)
    prob=1
    arr = []
    for i in ngramSentence:
        prev = tuple(i[:-1])
        curr = tuple([i[-1]])
        val = Witten_Bell(prev,curr,d)
        prob*=val
    return prob

def calculateProb_Kneser_Kney(sentence):
    avgPPLX=0
    perplexityArr=[]
    tokens = wordTokenize(sentence)
    ngramSentence = ngrams(tokens,4)
    prob=1
    arr = []
    for i in ngramSentence:
        prev = tuple(i[:-1])
        curr = tuple([i[-1]])
        val = Kneser_Ney(prev,curr,1,d,0.75)
        prob*=val
    return prob



def train_test_split(sentences):
    indices = list(np.random.choice(len(sentences),1000,replace=False))
    trainData=[]
    testData=[]
    for i in range(len(sentences)):
        if i in indices:
            testData.append(sentences[i])
        else:
            trainData.append(sentences[i])
    return trainData, testData



smoothningType = sys.argv[1]
pathToCorpus = sys.argv[2]
#===================================== ** Load data ** =========================================
file = open(pathToCorpus, "r")
sentences = file.readlines()
processedSentence=sentenceTokenizer(sentences,4)
tokens = [wordTokenize(sentence) for sentence in sentences]
d, cache = createDict(processedSentence)
sumOfUnigramValues = np.sum(list(d[1].values()))

sentence = str(input("Enter input sentence:"))


if smoothningType=='k':
    processedSentence=sentenceTokenizer([sentence],4)
    (pplx) = calculateProb_Kneser_Kney(processedSentence[0])
    print("Probability:",pplx)
    
elif smoothningType=='w':
    processedSentence=sentenceTokenizer([sentence],4)
    (pplx) = calculateProb_Witten_Bell(processedSentence[0])
    print("Probability:",pplx)
else:
    print("Give valid smoothning type!")
#print("PS: ",perplexity_kneser_kneys(ps))


# ================================ ** Generating o/p file code ** =============================================

filename1 = "2022201053_LM1_train-perplexity.txt"
filename2 = "2022201053_LM1_test-perplexity.txt"
filename3 = "2022201053_LM2_train-perplexity.txt"
filename4 = "2022201053_LM2_test-perplexity.txt"
filename5 = "2022201053_LM3_train-perplexity.txt"
filename6 = "2022201053_LM3_test-perplexity.txt"
filename7 = "2022201053_LM4_train-perplexity.txt"
filename8 = "2022201053_LM4_test-perplexity.txt"

#======================================= ** Corpus 1 ** ======================================================

#Corpus 1
file = open('Pride and Prejudice - Jane Austen.txt', "r")
sentences = file.readlines()
#lines = nltk.tokenize.sent_tokenize(lines)
processedSentence=sentenceTokenizer(sentences,4)
tokens = [wordTokenize(sentence) for sentence in sentences]
train, test = train_test_split(processedSentence)
d, cache = createDict(train)
sumOfUnigramValues = np.sum(list(d[1].values()))
(train_avgpplx, train_pplxarr) = perplexity_kneser_kneys(train)
(test_avgpplx, test_pplxarr) = perplexity_kneser_kneys(test)

#Train corpus1 Kneser Kney LM1
f = open(filename1, "w")
f.write(str(train_avgpplx)+"\n")
s=""
for i in range(len(train_pplxarr)):
    s+= train[i]+"  "+str(train_pplxarr[i])+"\n"
f.write(s)
f.close()

#Test corpus1 Kneser Kney LM2
f = open(filename2, "w")
f.write(str(test_avgpplx)+"\n")
s=""
for i in range(len(test_pplxarr)):
    s+= test[i]+"  "+str(test_pplxarr[i])+"\n"
f.write(s)
f.close()

print("K Train:",train_avgpplx)
print("K Test:",test_avgpplx)

(train_avgpplx, train_pplxarr) = perplexity_Witten_Bell(train)
(test_avgpplx, test_pplxarr) = perplexity_Witten_Bell(test)

#Train corpus1 Witten Bell LM3
f = open(filename3, "w")
f.write(str(train_avgpplx)+"\n")
s=""
for i in range(len(train_pplxarr)):
    s+= train[i]+"  "+str(train_pplxarr[i])+"\n"
f.write(s)
f.close()

#Test corpus1 Witten Bell LM4
f = open(filename4, "w")
f.write(str(test_avgpplx)+"\n")
s=""
for i in range(len(test_pplxarr)):
    s+= test[i]+"  "+str(test_pplxarr[i])+"\n"
f.write(s)
f.close()


#================================ ** Corpus 2 ** ======================================================

#Corpus 2
file = open('Ulysses - James Joyce.txt', "r")
sentences = file.readlines()
#lines = nltk.tokenize.sent_tokenize(lines)
processedSentence=sentenceTokenizer(sentences,4)
tokens = [wordTokenize(sentence) for sentence in sentences]
train, test = train_test_split(processedSentence)
d, cache = createDict(train)
sumOfUnigramValues = np.sum(list(d[1].values()))
(train_avgpplx, train_pplxarr) = perplexity_kneser_kneys(train)
(test_avgpplx, test_pplxarr) = perplexity_kneser_kneys(test)

#Train corpus2 Kneser Kney LM3 train
f = open(filename5, "w")
f.write(str(train_avgpplx)+"\n")
s=""
for i in range(len(train_pplxarr)):
    s+= train[i]+"  "+str(train_pplxarr[i])+"\n"
f.write(s)
f.close()

#Test corpus2 Kneser Kney LM3 test
f = open(filename6, "w")
f.write(str(test_avgpplx)+"\n")
s=""
for i in range(len(test_pplxarr)):
    s+= test[i]+"  "+str(test_pplxarr[i])+"\n"
f.write(s)
f.close()


(train_avgpplx, train_pplxarr) = perplexity_Witten_Bell(train)
(test_avgpplx, test_pplxarr) = perplexity_Witten_Bell(test)

#Train corpus2 Witten Bell LM4 train
f = open(filename7, "w")
f.write(str(train_avgpplx)+"\n")
s=""
for i in range(len(train_pplxarr)):
    s+= train[i]+"  "+str(train_pplxarr[i])+"\n"
f.write(s)
f.close()

#Test corpus2 Witten Bell LM4 test
f = open(filename8, "w")
f.write(str(test_avgpplx)+"\n")
s=""
for i in range(len(test_pplxarr)):
    s+= test[i]+"  "+str(test_pplxarr[i])+"\n"
f.write(s)
f.close()
