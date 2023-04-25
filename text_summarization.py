import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

text = '''ChatGPT is a member of the generative pre-trained transformer (GPT) family of language models. It was fine-tuned (an approach to transfer learning[6]) over an improved version of OpenAI's GPT-3 known as "GPT 3.5".[7] The fine-tuning process leveraged both supervised learning as well as reinforcement learning in a process called reinforcement learning from human feedback (RLHF).[8][9] Both approaches used human trainers to improve the model's performance. In the case of supervised learning, the model was provided with conversations in which the trainers played both sides: the user and the AI assistant. In the reinforcement learning step, human trainers first ranked responses that the model had created in a previous conversation.[10] These rankings were used to create 'reward models' that the model was further fine-tuned on using several iterations of Proximal Policy Optimization (PPO).[8][11] Proximal Policy Optimization algorithms present a cost-effective benefit to trust region policy optimization algorithms; they negate many of the computationally expensive operations with faster performance.[12][13] The models were trained in collaboration with Microsoft on their Azure supercomputing infrastructure, using Nvidia GPUs, "supercomputer developed for OpenAI is a single system with more than 285,000 CPU cores, 10,000 GPUs and 400 gigabits per second of network connectivity for each GPU server".[14]

In addition, OpenAI continues to gather data from ChatGPT users that could be used to further train and fine-tune ChatGPT. Users can upvote or downvote responses they receive from ChatGPT and fill out a text field with additional feedback.[15][16]


'''
def summarizer(rawdocs):

    stopwords = list(STOP_WORDS)
    #print(stopwords)
    nlp = spacy.load('en_core_web_sm')
    doc=nlp(rawdocs)
    tokens = [token.text for token in doc]
    #print(tokens)
    word_freq={}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_freq.keys():
                word_freq[word.text]=1
            else:
                word_freq[word.text] +=1
    #print(word_freq)
    max_freq = max(word_freq.values())
    #print(max_freq)           

    for word in word_freq.keys():
        word_freq[word] = word_freq[word]/max_freq
        
    #print(word_freq)

    sent_token = [sent for sent in doc.sents]
    # print(sent_token)

    sent_scores = {}
    for sent in sent_token:
        for word in sent:
            if word.text in word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_freq[word.text]
                else:
                    sent_scores[sent] += word_freq[word.text]
                    
    #print(sent_scores)

    select_len = int(len(sent_token)*0.3)
    summary = nlargest(select_len,sent_scores, key=sent_scores.get)
    # print(summary)
    final_sum = [word.text for word in summary]
    summary = ' '.join(final_sum)
    # print(text)
    # print(summary)
    # print("Length of original text",len(text.split(' ')))
    # print("Length of summary text",len(summary.split(' ')))
    return summary,doc,len(rawdocs.split(' ')), len(summary.split(' '))