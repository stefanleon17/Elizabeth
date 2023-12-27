import json
import random
import re
from nltk.chat.util import reflections, Chat

# importing emotion analysis and NER functionalities
from emotion_analysis import emotion_chat
from named_entity_recognition import extract_name
import eliza_replies

# Switch between "hello" sentences, which are processed differently, and "feel" sentences
def classifySentence(input_text, user=""):
    if "my name" in input_text.lower() or "hello" in input_text.lower():
        sentence_type = "hello"
    else:
        sentence_type = "feel"
    if sentence_type == "hello":
        name = extract_name(input_text)
        if name != "":
            person = name
            answer(input_text=input_text, person=name, sentence="hello")
        else:
            answer(input_text=input_text,category="clarification", sentence="hello")
    elif sentence_type == "feel":
        answer(input_text=input_text)

# Chatbot matching rules
replies = eliza_replies.eliza_pairs

chatbot = Chat(replies, reflections)

# Main chatting function
# computes and prints the chatbot's reply
def answer(input_text, category="response", sentence="feel", person=""):
    reply = ""
    if category == "clarification":
        if sentence == "feel":
            reply = random.choice(["I don't understand that sentence, can you please rephrase it?",
                           "Could you try expressing that in different words?"])
        elif sentence == "hello":
            reply = "I didn't understand that. Could you tell me your name again?"
    elif category == "response":
        if sentence == "hello":
            if eliza_replies.person_exists(person):
                reply = chatbot.respond("Hello")
                reply = eliza_replies.additional_processing("Hello", reply)
            else:
                reply = "Nice to meet you, " + person + "! How are you today?"
        elif sentence == "feel":
            emotion_reply = emotion_chat(input_text)
            eliza_reply = chatbot.respond(input_text)
            reply = eliza_reply + " " + emotion_reply
            reply = eliza_replies.additional_processing(input_text, reply)
    print(reply)

def user_interface():
    print("\n\n-------In order to quit, write 'quit' at any point-------\n\n\n\n"
          "Elizabeth: Hello! I am Elizabeth, your therapist chatbot! What's your name?")
    while True:
        print(">", end=" "),
        input_text = input()
        # Implementing an exit command
        if input_text == "quit":
            break
        print("Elizabeth: ", end=" "),
        classifySentence(input_text)


user_interface()
