# COMP 3074 - Human-AI Interaction Coursework 1
# Tan Lik Wei 20208762

# Import Libraries
import random
import pandas as pd
import numpy as np 
import nltk 
import string
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Only run these on the first run
# To use Punkt tokenizer
nltk.download('punkt')
# To use wordnet dictionary
nltk.download('wordnet')
nltk.download('omw-1.4')

# Read Corpus
qa_df = pd.read_csv('./QA_Dataset.csv')
qa_ques = list(qa_df['Question'])
smalltalk_df = pd.read_csv('./SmallTalk_Dataset.csv', encoding = 'windows-1252')
smalltalk_ques = list(smalltalk_df['Question'])

# Greeting Corpus
greeting_inputs = ["Greetings!","Hello!","Hi.","Greetings.","Hi there!","Hey.","Wassup!", "Hey there!", "Yo."]
greeting_responses = ["Hi 👋", "Hey 👋", "Hey there!","*nods*", "Hi there 👋", "Hello 👋", "Nice to meet you"]
# Time and Date Questions Corpus
date_time_inputs = ["Good Morning","Good Evening","Good Afternoon","Good Night","Morning.","Afternoon.","Evening.","Night.","What is the time now?","What time is it now?", "can you tell me the date today","What date is it today?","What is today's date?", "What date is today?","can you tell me the time", "what is the date today?", "now what time?", "today's date?"]
# Name Corpus
uname_inputs = ["What is my name?", "Do you remember my name?", "Do you remember me?", "Do you know my name?", "Who am I?", "Tell me who am I", "Say my name", "Please state my name"]

# Preprocessing
# WordNet is a semantically-oriented dictionary of English from NLTK
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def lemTokens(tokens):
    return[lemmatizer.lemmatize(token) for token in tokens]
remove_punctuation_dict = dict((ord(punct), None) for punct in string.punctuation)

def lemNormalize(text):
    return lemTokens(word_tokenize(text.translate(remove_punctuation_dict).lower()))

# Question Answering Response
def qa_response(user_in, token):  
    # print("Entered the QA dataset")
    TfidfVec = TfidfVectorizer(tokenizer = lemNormalize, stop_words = 'english')
    tfidf = TfidfVec.fit_transform(token)
    # print(tfidf)
    tfidf_query = TfidfVec.transform([user_in]).toarray()
    vals = cosine_similarity(tfidf_query, tfidf)
    idx = np.argmax(vals)
    # print(idx)

    bot_response = ''
    if vals.max() > 0:
        # print("Fetching answer from dataset...")
        return qa_df['Answer'][idx]
    else:
        bot_response = bot_response + "Sorry " + username + ", I couldn't understand you 😞. Can you describe more about it?"
        return bot_response

# Small Talk Response
def st_response(user_in, token2):
    # print("Searching the SmallTalk dataset")
    TfidfVec = TfidfVectorizer(tokenizer = lemNormalize)
    tfidf = TfidfVec.fit_transform(token2)
    tfidf_query = TfidfVec.transform([user_in]).toarray()
    vals = cosine_similarity(tfidf_query, tfidf)
    idx = np.argmax(vals)

    bot_response = ''
    if vals.max() > 0:
        return smalltalk_df['Answer'][idx]
    else:
        bot_response = bot_response + "Sorry " + username + ", I couldn't understand you 😞. Can you describe more about it?"
        return bot_response

# Similarity Function
def similarity(token, query):
    TfidfVec = TfidfVectorizer(tokenizer = lemNormalize, min_df = 0.01)
    tfidf = TfidfVec.fit_transform(token).toarray()
    tfidf_query = TfidfVec.transform([query]).toarray()
    vals = cosine_similarity(tfidf_query, tfidf)
    return vals

# Intent Matching
def intent_matching(user_response):
    ans_val = similarity(qa_ques, user_response).max()
    smalltalk_val = similarity(smalltalk_ques, user_response).max()
    greeting_val = similarity(greeting_inputs, user_response).max()
    date_time_val = similarity(date_time_inputs, user_response).max()
    username_val = similarity(uname_inputs, user_response).max()

    val_arr = [ans_val, smalltalk_val, greeting_val, date_time_val, username_val]
    print("The maximum similarity scores in each category are ")
    print(val_arr)

    if max(val_arr) < 0.5:
        return qa_response(user_response, qa_ques)
    else:
        idx = np.argsort(val_arr, None)[-1]
        # print(idx)

        if idx == 0:
            return qa_response(user_response, qa_ques)
        elif idx == 1:
            return st_response(user_response, smalltalk_ques)
        elif idx == 2:
            return random.choice(greeting_responses)
        elif idx == 3:
            current_time = datetime.now()
            if 5 <= current_time.hour < 12:
                print('Good morning!')
            elif 12 <= current_time.hour < 17:
                print('Good afternoon!')
            else:
                print('Good evening!')           
            return "The current date and time is " + current_time.strftime("%Y-%m-%d %H:%M:%S") 
        elif idx == 4:
            return "Your name is " + username + "😄"


# Chatbot Main Interface
# Identity Management
print('')
print("Rakan: Hey there, I'm Rakan 😎. May I know your name?")
noname = True
while(noname == True):
    print("> ", end = "")
    username = input()
    if(username == ""):
        print('')
        print("Rakan: Sorry, I didn't manage to get your name. Can you reenter your name please?")
    else:
        noname = False

active_flag = True
print('')
print("Rakan: Hi " + username + ", how may I help you? If you would like to quit, type \"Bye\"!")
while(active_flag == True):
    print("> ", end = "")
    user_response = input()
    user_response = user_response.lower().translate(remove_punctuation_dict)

    if user_response != 'bye':
        if user_response in ['thank you', 'thanks', 'thanks a lot']: 
            print('')
            print("Rakan: You are welcome ✌️. What else can I help you with?")
        else:
            print('Rakan: ', end = "")
            print(intent_matching(user_response))
            print('')
    else:
        active_flag = False
        print('')
        print("Rakan: Goodbye " + username + ", it was nice talking to you today! See you soon 👋")