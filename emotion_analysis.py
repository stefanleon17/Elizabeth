# Implements emotion analysis functionalities
# transformer source: https://huggingface.co/mrm8488/t5-base-finetuned-emotion?text=I%27m+feeling+odd

from eliza_replies import additional_processing
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Implement emotion analysis pipeline
emotion_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
emotion_model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-emotion")

# Translating emotion analysis into meaningful sentences
emotion_replies = {
    "sadness": ["I can see that makes you sad.",
                "That must make you sad.",
                ],
    "fear": ["That sounds scary.",
             "I would be afraid, too.",
             ],
    "disgust": ["You sound disgusted by that.",
                "It sounds like that disgusts you.",
                ],
    "anger": ["That must make you angry.",
              "It sounds like that makes you angry.",
              "You sound angry about that.",
              ],
    "joy": ["It seems like that makes you happy.",
            "You seem to enjoy that."],
    "surprise": ["You seem surprised by that."],
    "love": ["You seem charmed!",
             "You seem to enjoy that."]
}

# keywords to track whether the user is talking about their feeling
emotion_keywords = ["sad","sadness", "happy", "happiness", "anger", "angry", "joy", "joyous", "fear", "afraid",
                    "shocked", "surprised", "cheerful", "disgusted", "furious", "depressed", "in love"]

def get_emotion(text):
    input_ids = emotion_tokenizer.encode(text + '</s>', return_tensors='pt')

    output = emotion_model.generate(input_ids=input_ids, max_length=2)

    results = [emotion_tokenizer.decode(ids) for ids in output]
    label = results[0]
    label = label[6:]
    return label

def emotion_chat(text):
    emotion = get_emotion(text)

    if any(word in text for word in emotion_keywords):
        emotion_reply = ""
    elif additional_processing(text) == "add_emotion":
        emotion_reply = random.choice(emotion_replies[emotion])
    elif add_emotion(probability):
        emotion_reply = random.choice(emotion_replies[emotion])
    else:
        emotion_reply = ""
    return emotion_reply

# Avoid over-saturating dialogue with emotion analysis
# by stochastically omitting some of it
probability = 70
def add_emotion(probability):
    choice = random.randint(0,100)
    if choice <= probability:
        probability -= 10
        return True
    else:
        probability += 10
        return False


