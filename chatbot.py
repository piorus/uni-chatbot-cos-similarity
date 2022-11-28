# write a chatbot that reacts on specific dialog [[Question, Answer], ...]
conversation = [["Hello", "Hi"], ["How are you?", "I am fine"], ["What is your name?", "My name is HAL"]]

# you can follow:
# 1. create TFIDF vectors for each question (try without stop words first)
# 2. write REPL that reads question, convert it to TFIDF vector, compute cos similarity, 
# finds best match, and returns answer to the best-matched question

# This is what simple chatbot do, therfore completing this task you will learn 
# how to write your own chatbot from scratch

# prepare corpus, and answers

corpus = list(zip(*conversation))[0]
print('corpus = ', corpus)

answers = list(zip(*conversation))[1]
print('answers = ', answers)

# create TFIDF vectors for each question (without stop words)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# write REPL that reads question

import numpy as np

def repl(query):
    Q = vectorizer.transform([query])

    # convert question to TFIDF vector

    Q_vec = np.array(Q.todense().copy())[0, :]
    X_vec = np.array(X.todense().copy())

    # compute cos similarity

    def cos_sim(v1, v2):
        costheta = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
        return costheta

    print("Compute cos_sim")

    similarities = [cos_sim(Q_vec, X_vec[i]) for i in range(len(corpus))]
    print('similarities = ', similarities)

    # find the best match

    index_max = max(range(len(similarities)), key=similarities.__getitem__) # ref: https://stackoverflow.com/a/11825864/12036989

    # return answer to the best-matched question

    print(answers[index_max])

while True:
    query = input("> ")
    if query != 'q': repl(query)
    else: break
