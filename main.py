import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow
import tflearn
import random
import json
import pickle

stemmer = LancasterStemmer()

with open("json_file/intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    # preprocess the data
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            print(wrds)
            words.extend(wrds)
            print("words", words)
            docs_x.append(wrds)
            print("docs_x", docs_x)
            docs_y.append(intent["tag"])
            print("docs_y", docs_y, "\n")

        if intent["tag"] not in labels:
            labels.append(intent["tag"])
            print("labels", labels)

    print("length if docs_x: ", len(docs_x))
    print("length if docs_y: ", len(docs_y))

    print("words before stemmer: ", words)
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    print("length if words: ", len(words))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for index, doc in enumerate(docs_x):
        bag = []
        print("doc: ", doc)
        wrds = [stemmer.stem(w) for w in doc]
        print("wrds: ", wrds)
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = list(out_empty)
        output_row[labels.index(docs_y[index])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# save the data in order to load later
# training the model]
tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 13)
net = tflearn.fully_connected(net, 13)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
'''
model = keras.Sequential()
embedding_layer = keras.layers.Embedding(len(training[0]), 8)
model.add(embedding_layer)
LSTM_layer = keras.layers.LSTM(10)
model.add(LSTM_layer)
output_layer = keras.layers.Dense(len(output[0]), activation="softmax")
model.add(output_layer)
model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
'''

# try:
#     keras.models.load_model("model.keras")
# except:
# save the model in order to load later
# try:
#     keras.models.load_model("model.keras")
# except:
try:
    model.load("../model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=10, show_metric=True)
    model.save("../model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for s_word in s_words:
        for i,w in enumerate(words):
            if w == s_word:
                bag[i] = 1

    return np.array(bag)

def chat():
    print("Start talking with the Josey!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = np.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]

            print(random.choice(responses))
        else:
            print("I do not get that, please speak human language, dumbass")
chat()
