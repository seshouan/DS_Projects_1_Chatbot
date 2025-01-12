import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from tensorflow.keras.models import load_model
model = load_model('chatbot_model.keras')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern (split words into an array)
    sentence_words = nltk.word_tokenize(sentence)
    # lemmatize each word (reduce to base form)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array of numbers (1 if word in sentence, else 0)
def bag_of_words(sentence, words, show_details=True):
    # tokenize patterns
    sentence_words = clean_up_sentence(sentence)
    # obtain the bag of words (the vocabulary matrix)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
                if show_details:
                    print(f'bag contains: {word}')
    return(np.array(bag))

def predict_class(sentence):
    # filter below threshold predictions
    p = bag_of_words(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort strength probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag']==tag):
            result = random.choice(i['responses'])
            break
    return result

# create the tkinter GUI
import tkinter
from tkinter import *

def send():
    msg = EntryBox.get('1.0', 'end-1c').strip()
    EntryBox.delete('0.0', END)

    if msg != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, 'You: ' + msg + '\n\n')
        ChatBox.config(foreground='#446665', font=('Verdana', 12))
        
        ints = predict_class(msg)
        res = getResponse(ints, intents)

        ChatBox.insert(END, 'Bot: ' + res + '\n\n')

        ChatBox.config(state = DISABLED)
        ChatBox.yview(END)

root = Tk()
root.title('Chatbot')
root.geometry('400x500')
root.resizable(width=False, height=False)

# create chat window
ChatBox = Text(root, bd=0, bg='white', height='8', width='50', font='Arial',)

ChatBox.config(state=DISABLED)

# bind scrollbar to chat window
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor='heart')
ChatBox['yscrollcommand'] = scrollbar.set

# create a send message button
SendButton = Button(root, font=('Verdana', 12, 'bold'), text='Send', width='12', height='5', bd=0, bg='#f9a602', activebackground='#3c9d9b', fg='#000000', command=send)

# create an input box
EntryBox = Text(root, bd=0, bg='white', width='29', height='5', font='Arial', fg='#000000')
# EntryBox.bind('<Return>', send)

# arrange all components
scrollbar.place(x=376, y=6, height=386)
ChatBox.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

root.mainloop()