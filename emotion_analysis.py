# Implements emotion analysis functionalities
# transformer source: https://huggingface.co/bdotloh/distilbert-base-uncased-empathetic-dialogues-context

from eliza_replies import pattern_match
import random
from transformers import pipeline

# Implement emotion analysis pipeline
emotion_pipeline = pipeline("text-classification", model="bdotloh/distilbert-base-uncased-empathetic-dialogues-context")

# Keywords to track whether the user is talking about their feeling
emotion_keywords = ["sad","sadness", "happy", "happiness", "anger", "angry", "joy", "joyous", "fear", "afraid",
                    "shocked", "surprised", "cheerful", "disgusted", "furious", "depressed", "in love"]


def emotion_chat(text):
    emotion = emotion_pipeline(text)[0]["label"]

    if any(word in text for word in emotion_keywords) or emotion in text:
        # if the user is already talking about their emotions,
        # there is no need to prompt them to do so
        emotion_reply = ""
    elif pattern_match(text, 34) or pattern_match(text, 35) or pattern_match(text, 36) or pattern_match(text, 40) or pattern_match(text, 41):
        # Matches with a high risk of ending the conversation
        # prompt the user with insight into their own feelings
        # to keep the conversation going
        emotion_reply = random.choice(["Do you feel " + emotion + " ?",
                                       "Would you say you feel " + emotion + " ?",
                                       "Are you " + emotion + " ?"])
    elif add_emotion(probability):
        # Stochastically prompt the user to talk about their emotions
        emotion_reply = random.choice(["Do you feel " + emotion + " ?",
                                       "Would you say you feel " + emotion + " ?",
                                       "Are you " + emotion + " ?"])
    else:
        emotion_reply = ""
    return emotion_reply

# Avoid over-saturating dialogue with emotion analysis
# by stochastically omitting some of it
probability = 40
def add_emotion(probability):
    choice = random.randint(0,100)
    if choice <= probability:
        probability -= 10
        return True
    else:
        probability += 10
        return False


