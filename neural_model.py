import re
import numpy as np
import sys
import keras
from keras.utils import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential




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


class DataFormat(keras.utils.Sequence):
    def __init__(self, sentences,maxLen, batch_size=32, shuffle=True):
        self.sentences = sentences
        self.indexes = np.arange(len(sentences))
        self.batch_size = batch_size
        self.shuffle = True

        
        self.d = self.createVocab(sentences)
        self.encodedSentences = self.encodingIntoSentences(sentences)
        self.maxLen = int(maxLen)
        self.on_epoch_end()

    def createVocab(self,sentences):
      di={}
      count=2
      di['<UNK>']=1
      for sent in sentences:
        for word in sent.split():
          if word not in di:
            di[word]=count
            count+=1
      
      return di

    def encodingIntoSentences(self,sentences):
      tokenized = [i.split() for i in sentences]
      encoded=[]
      for i in tokenized:
        li=[]
        for word in i:
          if word in self.d.keys():
            li.append(self.d[word])
          else:
            li.append(self.d['<UNK>'])
        encoded.append(li)
      return encoded

    # def averageLength(self,sentences):
    #   l=0
    #   i=0
    #   for sent in sentences:
    #     l+= len(sent)
    #     i+=1
    #   return l/i

    def getInputFormat(self,sentences,maxLen):
      x_train=[]
      y_train=[]
      for j in sentences:
        X=[]
        Y=[]
        for i in range(len(j)):
          x_temp = pad_sequences([j[:i]],maxlen=maxLen - 1,padding='pre')[0]
          X.append(x_temp)
          Y.append(j[i])
        x_train.append(X)
        y_train.append(Y)
      return x_train, y_train


    def __len__(self):
        return int(np.floor(len(self.sentences) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        encodedSentences = [self.encodedSentences[k] for k in indexes]
        x = []
        y = []
        for seq in encodedSentences:
            xDelta, yDelta = self.getInputFormat(seq, self.maxLen)
            x += xDelta
            y += yDelta
        x = np.array(x)
        for i in range(len(y)):
          y[i]=y[i]-1
        y = np.array(y)
        y = np.eye(len(self.d))[y]
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


class DataGenerate(keras.utils.Sequence):
    def __init__(self, sentences,maxLen, batch_size=32, shuffle=True):
        self.sentences = sentences
        self.indexes = np.arange(len(sentences))
        self.batch_size = batch_size
        self.shuffle = True

        
        self.d = self.createVocab(sentences)
        self.encodedSentences = self.encodingIntoSentences(sentences)
        self.maxLen = int(maxLen)
        self.on_epoch_end()

    def createVocab(self,sentences):
      di={}
      count=2
      di['<UNK>']=1
      for sent in sentences:
        for word in sent.split():
          if word not in di:
            di[word]=count
            count+=1
      
      return di

    def encodingIntoSentences(self,sentences):
      tokenized = [i.split() for i in sentences]
      encoded=[]
      for i in tokenized:
        li=[]
        for word in i:
          if word in self.d.keys():
            li.append(self.d[word])
          else:
            li.append(self.d['<UNK>'])
        encoded.append(li)
      return encoded

    # def averageLength(self,sentences):
    #   l=0
    #   i=0
    #   for sent in sentences:
    #     l+= len(sent)
    #     i+=1
    #   return l/i

    def getInputFormat(self,sentences,maxLen):
      x_train=[]
      y_train=[]
      
      for i in range(len(sentences)):
        x_temp = pad_sequences([sentences[:i]],maxlen=maxLen - 1,padding='pre')[0]
        x_train.append(x_temp)
        y_train.append(sentences[i])
      return x_train, y_train


    def __len__(self):
        return int(np.floor(len(self.sentences) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        encodedSentences = [self.encodedSentences[k] for k in indexes]
        x = []
        y = []
        for seq in encodedSentences:
            xDelta, yDelta = self.getInputFormat(seq, self.maxLen)
            x += xDelta
            y += yDelta
        x = np.array(x)
        for i in range(len(y)):
          y[i]=y[i]-1
        y = np.array(y)
        y = np.eye(len(self.d))[y]

        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

pathToCorpus = sys.argv[1]
modelPath = sys.argv[2]
#Load data
file = open(pathToCorpus, "r")
sentences = file.readlines()
processedSentence=sentenceTokenizer(sentences,4)
tokens = [wordTokenize(sentence) for sentence in sentences]
if modelPath=="ulyssus.h5":
    generator = DataGenerate(processedSentence,68,batch_size=4,shuffle=True)
    vocab_size = len(generator.d)
    model = keras.models.load_model(modelPath)
elif modelPath=="prideModel.h5":
    generator = DataGenerate(processedSentence,128,batch_size=4,shuffle=True)
    vocab_size = len(generator.d)
    model = keras.models.load_model(modelPath)

#================================= ** Training Model Code ** =======================================
model = Sequential()
model.add(Embedding(input_dim=vocab_size + 1,output_dim=5,input_length=generator.maxLen))
model.add(LSTM(10))
model.add(Dense(vocab_size, activation='softmax'))
model.compile('rmsprop', 'categorical_crossentropy')
model.fit_generator(generator, epochs=5)

sentence = str(input("Enter input sentence:"))

tok = generator.encodingIntoSentences([sentence])[0]

x_test, y_test = generator.getInputFormat(tok, generator.maxLen)

x_test = np.array(x_test)
for i in range(len(y_test)):
  y_test[i]-=1

y_test = np.array(y_test)

predictions = model.predict(x_test)
inverseD = {v: k for k, v in generator.d.items()}
log_p_sentence = 0
for i in range(len(predictions)):
    word = inverseD[y_test[i] + 1]
    history = ' '.join([inverseD[word] for word in x_test[i, :] if word != 0])
    prob_word = predictions[i][y_test[i]]
    log_p_sentence += np.log(prob_word)
print("Probability:",np.exp(log_p_sentence))

filename5 ="2022201053_LM5_train-perplexity"
filename6 ="2022201053_LM5_test-perplexity"
filename7 ="2022201053_LM6_train-perplexity"
filename8 ="2022201053_LM6_test-perplexity"
#===================================== ** Corpus 1 ** ===========================================
file = open("ppja.txt", "r")
lines = file.readlines()
processedSentence=sentenceTokenizer(lines,4)

train , test = train_test_split(processedSentence)
generator = DataFormat(train,127,batch_size=4,shuffle=True)
vocab_size = len(generator.d)
#model = Sequential()
# model.add(Embedding(input_dim=vocab_size + 1,output_dim=5,input_length=generator.maxLen))
# model.add(LSTM(10))
# model.add(Dense(vocab_size, activation='softmax'))
# model.compile('rmsprop', 'categorical_crossentropy')
# model.fit_generator(generator, epochs=5)
model = keras.models.load_model('prideModel.h5')

# sentence = "I am a woman and I love boating"

#test
tok = generator.encodingIntoSentences(test)
x_test, y_test = generator.getInputFormat(tok, generator.maxLen+1)
p_pred=[]
for k in range(len(x_test)):
  x = np.array(x_test[k])
  y = np.array(y_test[k]) - 1 
  p = model.predict(x)
  p_pred.append(p)
vocab_inv = {v: k for k, v in generator.d.items()}
s=""

pplxArr=[]
for j in range(len(p_pred)):
  log_p_sentence = 0
  for i, prob in enumerate(p_pred[j]):
      word = vocab_inv[y_test[j][i] + 1]  # Index 0 from vocab is reserved to <PAD>
      history = ' '.join([vocab_inv[w] for w in x_test[j][i][:] if w != 0])
      prob_word = prob[y_test[j][i]]
      log_p_sentence += np.log(prob_word)
      #print('P(w={}|h={})={}'.format(word, history, prob_word))
  #print('Prob. sentence: {}'.format(np.exp(log_p_sentence)))
  pplx = perplexity(np.exp(log_p_sentence),len(test[j].split()))
  pplxArr.append(pplx)
  s+=test[j]+"  "+str(pplx)+" \n"
  print(j,pplx)

meanpplx = sum(pplxArr)/len(pplxArr)
s = str(meanpplx)+"\n"+s
f = open(filename6, "w")
f.write(s)
f.close()

#Train
file = open("ppja.txt", "r")
lines = file.readlines()
processedSentence=sentenceTokenizer(lines,4)

train , test = train_test_split(processedSentence)
generator = DataFormat(train,127,batch_size=4,shuffle=True)
vocab_size = len(generator.d)
#model = Sequential()
# model.add(Embedding(input_dim=vocab_size + 1,output_dim=5,input_length=generator.maxLen))
# model.add(LSTM(10))
# model.add(Dense(vocab_size, activation='softmax'))
# model.compile('rmsprop', 'categorical_crossentropy')
# model.fit_generator(generator, epochs=5)
model = keras.models.load_model('prideModel.h5')

# sentence = "I am a woman and I love boating"

#test
tok = generator.encodingIntoSentences(train)
x_test, y_test = generator.getInputFormat(tok, generator.maxLen+1)
p_pred=[]
for k in range(len(x_test)):
  x = np.array(x_test[k])
  y = np.array(y_test[k]) - 1 
  p = model.predict(x)
  p_pred.append(p)
vocab_inv = {v: k for k, v in generator.d.items()}
s=""
pplxArr=[]
for j in range(len(p_pred)):
  log_p_sentence = 0
  for i, prob in enumerate(p_pred[j]):
      word = vocab_inv[y_test[j][i]]  # Index 0 from vocab is reserved to <PAD>
      history = ' '.join([vocab_inv[w] for w in x_test[j][i][:] if w != 0])
      prob_word = prob[y_test[j][i]]
      log_p_sentence += np.log(prob_word)
      #print('P(w={}|h={})={}'.format(word, history, prob_word))
  #print('Prob. sentence: {}'.format(np.exp(log_p_sentence)))
  pplx = perplexity(np.exp(log_p_sentence),len(train[j].split()))
  pplxArr.append(pplx)
  s+=train[j]+"  "+str(pplx)+" \n"
  print(j,pplx)

meanpplx = sum(pplxArr)/len(pplxArr)
s = str(meanpplx)+"\n"+s
f = open(filename5, "w")
f.write(s)
f.close()


#======================================= ** Corpus 2 ** ==============================================

file = open("ujj.txt", "r")
lines = file.readlines()
processedSentence=sentenceTokenizer(lines,4)

train , test = train_test_split(processedSentence)
generator = DataFormat(train,68,batch_size=4,shuffle=True)
vocab_size = len(generator.d)
#model = Sequential()
# model.add(Embedding(input_dim=vocab_size + 1,output_dim=5,input_length=generator.maxLen))
# model.add(LSTM(10))
# model.add(Dense(vocab_size, activation='softmax'))
# model.compile('rmsprop', 'categorical_crossentropy')
# model.fit_generator(generator, epochs=5)
model = keras.models.load_model('ulyssus.h5')

# sentence = "I am a woman and I love boating"

#test
tok = generator.encodingIntoSentences(test)
x_test, y_test = generator.getInputFormat(tok, generator.maxLen+1)
p_pred=[]
for k in range(len(x_test)):
  x = np.array(x_test[k])
  y = np.array(y_test[k]) - 1 
  p = model.predict(x)
  p_pred.append(p)
vocab_inv = {v: k for k, v in generator.d.items()}
s=""
pplxArr=[]
for j in range(len(p_pred)):
  log_p_sentence = 0
  for i, prob in enumerate(p_pred[j]):
      word = vocab_inv[y_test[j][i]]  # Index 0 from vocab is reserved to <PAD>
      history = ' '.join([vocab_inv[w] for w in x_test[j][i][:] if w != 0])
      prob_word = prob[y_test[j][i]]
      log_p_sentence += np.log(prob_word)
      #print('P(w={}|h={})={}'.format(word, history, prob_word))
  #print('Prob. sentence: {}'.format(np.exp(log_p_sentence)))
  pplx = perplexity(np.exp(log_p_sentence),len(test[j].split()))
  pplxArr.append(pplx)
  s+=test[j]+"  "+str(pplx)+" \n"
  print(j,pplx)

meanpplx = sum(pplxArr)/len(pplxArr)
s = str(meanpplx)+"\n"+s
f = open(filename8, "w")
f.write(s)
f.close()


#================================= ** Corpus 2 ** =======================================

file = open("ujj.txt", "r")
lines = file.readlines()
processedSentence=sentenceTokenizer(lines,4)

train , test = train_test_split(processedSentence)
generator = DataFormat(train,68,batch_size=4,shuffle=True)
vocab_size = len(generator.d)
#model = Sequential()
# model.add(Embedding(input_dim=vocab_size + 1,output_dim=5,input_length=generator.maxLen))
# model.add(LSTM(10))
# model.add(Dense(vocab_size, activation='softmax'))
# model.compile('rmsprop', 'categorical_crossentropy')
# model.fit_generator(generator, epochs=5)
model = keras.models.load_model('ulyssus.h5')

# sentence = "I am a woman and I love boating"

#test
tok = generator.encodingIntoSentences(train)
x_test, y_test = generator.getInputFormat(tok, generator.maxLen+1)
p_pred=[]
for k in range(len(x_test)):
  x = np.array(x_test[k])
  y = np.array(y_test[k]) - 1 
  p = model.predict(x)
  p_pred.append(p)
vocab_inv = {v: k for k, v in generator.d.items()}
s=""
pplxArr=[]
for j in range(len(p_pred)):
  log_p_sentence = 0
  for i, prob in enumerate(p_pred[j]):
      word = vocab_inv[y_test[j][i]]  # Index 0 from vocab is reserved to <PAD>
      history = ' '.join([vocab_inv[w] for w in x_test[j][i][:] if w != 0])
      prob_word = prob[y_test[j][i]]
      log_p_sentence += np.log(prob_word)
      #print('P(w={}|h={})={}'.format(word, history, prob_word))
  #print('Prob. sentence: {}'.format(np.exp(log_p_sentence)))
  pplx = perplexity(np.exp(log_p_sentence),len(train[j].split()))
  pplxArr.append(pplx)
  s+=train[j]+"  "+str(pplx)+" \n"
  print(j,pplx)

meanpplx = sum(pplxArr)/len(pplxArr)
s = str(meanpplx)+"\n"+s
f = open(filename7, "w")
f.write(s)
f.close()
