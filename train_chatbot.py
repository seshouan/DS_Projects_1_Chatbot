# import required modules
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle

# import nltk resources
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('punkt')

# read in a json file
lemmatizer = WordNetLemmatizer()
intents_file = open('intents.json').read()
intents = json.loads(intents_file)

# preprocessing: tokenizer
words = []
classes = []
documents = []
ignore_letters = list('!?,.')

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        # add documents to the corpus
        documents.append((word, intent['tag']))
        # add to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(documents)

# preprocessing: lemmatizer

# lemmatize and lowercase each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# print number of documents, which combine patterns and intents
print(len(documents), 'documents')
# print classes as intents
print(len(classes), 'classes', classes)
# print all words in the vocabulary
print(len(words), 'unique lemmatized words', words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# convert input patterns to numbers

# create the training data
training = []
# create output array
output_empty = [0] * len(classes)
# use a bag of words for every sentence as the training set
for doc in documents:
    # init a bag of words
    bag = []
    # get the list of tokenized words for the pattern
    word_patterns = doc[0]
    # lemmatize each word
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # flag words that appear in the current pattern as 1s the rest as 0s
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # output 1s for the current tag and 0s otherwise
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
# shuffle the training set and turn it to a numpy array
random.shuffle(training)
training = np.array(training, dtype=object)
# create training inputs (patterns) and label (intent) sets
train_X = list(training[:,0])
train_y = list(training[:,1])
print('created training data')

# train the model with 3 layers deep, using dropout layers and the SGD optimizer; train 200 epochs

# define the deep neural net
model = Sequential()
model.add(Dense(128, input_shape=(len(train_X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# compile the model using SGD with Nesterov accelerated gradient
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# train and save the model
hist = model.fit(np.array(train_X), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print('created model')