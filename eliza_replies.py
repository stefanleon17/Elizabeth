# Code based off of the nltk implementation of the ELIZA chatbot
# https://www.nltk.org/_modules/nltk/chat/eliza.html

import json
import re
import string

from named_entity_recognition import extract_name

def read_JSON():
    with open("users.json", "r") as file:
        user_data = json.load(file)
    return user_data
# JSON write
def write_JSON(data):
    with open("users.json", "w") as file:
        json.dump(data, file, indent=2)

user_data = read_JSON()

class User:
    def __init__(self, name=""):
        self.name = ""
        self.mother = ""
        self.father = ""
        self.friend = ""
        self.sibling = ""

    def copy_from_json(self, user):
        self.name = user["name"]
        self.friend = user["friend"]
        self.sibling = user["sibling"]
        self.mother = user["mother"]
        self.father = user["father"]


user = User()

def person_exists(name):
    if name in user_data:
        user.copy_from_json(user_data[name])
        return True
    else:
        user.name = name
        return False

eliza_pairs = (
    (
        # Position 0
        # "hello" type sentence response
        r"Hello(.*)",
        (
            "Hello, name, I'm glad you could drop by today. Is something troubling you?",
            "Hi there, name, how are you today?",
            "Hello, name, how are you feeling today?",
        ),
    ),
    (
        # Position 1-5
        # Replies where we can exploit NER functionality
        r"(.*) friend (.*)",
        (
            "Tell me more about your friends.",
            "When you think of a friend, what comes to mind?",
            "Why don't you tell me about this friend?",
        ),
    ),
    (
        r"(.*) mother(.*)",
        (
            "Tell me more about your mother.",
            "What was your relationship with your mother like?",
            "How do you feel about your mother?",
            "How does this relate to your feelings today?",
            "Good family relations are important.",
        ),
    ),
    (
        r"(.*) father(.*)",
        (
            "Tell me more about your father.",
            "How did your father make you feel?",
            "How do you feel about your father?",
            "Does your relationship with your father relate to your feelings today?",
            "Do you have trouble showing affection with your family?",
        ),
    ),
    (
        r"(.*) sister(.*)",
        (
            "Tell me more about your sister.",
            "How did your sister make you feel?",
            "How do you feel about your sister?",
            "Does your relationship with your sister relate to your feelings today?",
            "Do you have trouble showing affection with your family?",
        ),
    ),
    (
        r"(.*) brother(.*)",
        (
            "Tell me more about your brother.",
            "How did your brother make you feel?",
            "How do you feel about your brother?",
            "Does your relationship with your brother relate to your feelings today?",
            "Do you have trouble showing affection with your family?",
        ),
    ),
    (
        # Position 6
        # Exit case
        r"quit",
        (
            "Thank you for talking with me.",
            "Good-bye.",
            "Thank you, that will be $150.  Have a good day!",
        ),
    ),
    (
        r"I need (.*)",
        (
            "Why do you need %1?",
            "Would it really help you to get %1?",
            "Are you sure you need %1?",
        ),
    ),
    (
        r"Why don\'t you (.*)",
        (
            "Do you really think I don't %1?",
            "Perhaps eventually I will %1.",
            "Do you really want me to %1?",
        ),
    ),
    (
        r"Why can\'t I (.*)",
        (
            "Do you think you should be able to %1?",
            "If you could %1, what would you do?",
            "Think about it -- why can't you %1?",
            "Have you really tried?",
        ),
    ),
    (
        r"I can\'t (.*)",
        (
            "How do you know you can't %1?",
            "Perhaps you could %1 if you tried.",
            "What would it take for you to %1?",
        ),
    ),
    (
        r"I am (.*)",
        (
            "Did you come to me because you are %1?",
            "How long have you been %1?",
            "How do you feel about being %1?",
        ),
    ),
    (
        r"I\'m (.*)",
        (
            "How does being %1 make you feel?",
            "Do you enjoy being %1?",
            "Why do you tell me you're %1?",
            "Why do you think you're %1?",
        ),
    ),
    (
        r"Are you (.*)",
        (
            "Why does it matter whether I am %1?",
            "Would you prefer it if I were not %1?",
            "Perhaps you believe I am %1.",
            "I may be %1 -- what do you think?",
        ),
    ),
    (
        r"What (.*)",
        (
            "Why do you ask?",
            "How would an answer to that help you?",
            "What do you think?",
        ),
    ),
    (
        r"(.*) sorry (.*)",
        (
            "There are many times when no apology is needed.",
            "What feelings do you have when you apologize?",
        ),
    ),
    (
        r"(.*) computer(.*)",
        (
            "Are you really talking about me?",
            "Does it seem strange to talk to a computer?",
            "How do computers make you feel?",
            "Do you feel threatened by computers?",
        ),
    ),
    (
        r"Is it (.*)",
        (
            "Do you think it is %1?",
            "Perhaps it is %1 -- what do you think?",
            "If it were %1, what would you do?",
        ),
    ),
    (
        r"It is (.*)",
        (
            "You seem very certain. If I told you that it probably isn't %1, how would you feel?",
        ),
    ),
    (
        r"Can you (.*)",
        (
            "What makes you think I can't %1?",
            "If I could %1, then what?",
            "Why do you ask if I can %1?",
        ),
    ),
    (
        r"Can I (.*)",
        (
            "Perhaps you don't want to %1.",
            "Do you want to be able to %1?",
            "If you could %1, would you?",
        ),
    ),
    (
        r"You are (.*)",
        (
            "Why do you think I am %1?",
            "Does it please you to think that I'm %1?",
            "Perhaps you would like me to be %1.",
            "Perhaps you're really talking about yourself?",
        ),
    ),
    (
        r"You\'re (.*)",
        (
            "Why do you say I am %1?",
            "Why do you think I am %1?",
            "Are we talking about you, or me?",
        ),
    ),
    (
        r"I don\'t (.*)",
        (
            "Don't you really %1?",
            "Why don't you %1?",
            "Do you want to %1?"
        ),
    ),
    (
        r"I feel (.*)",
        (
            "Tell me more about these feelings.",
            "Do you often feel %1?",
            "When do you usually feel %1?",
            "When you feel %1, what do you do?",
        ),

    ),
    (
        r"I am feeling (.*)",
        (
            "Tell me more about these feelings.",
            "Do you often feel %1?",
            "When do you usually feel %1?",
            "When you feel %1, what do you do?",
        ),

    ),
    (
        r"I have (.*)",
        (
            "Why do you tell me that you have %1?",
            "Now that you have %1, what will you do next?",
        ),
    ),
    (
        r"I would (.*)",
        (
            "Could you explain why you would %1?",
            "Why would you %1?",
            "Who else knows that you would %1?",
        ),
    ),
    (
        r"Is there (.*)",
        (
            "Do you think there is %1?",
            "Would you like there to be %1?",
        ),
    ),
    (
        r"My (.*)",
        (
            "I see, your %1.",
            "Why do you say that your %1?",
            "When your %1, how do you feel?",
        ),
    ),
    (
        r"You (.*)",
        (
            "We should be discussing you, not me.",
            "Why do you say that about me?",
            "Why do you care whether I %1?",
        ),
    ),
    (
        r"Why (.*)",
        (
             "Why don't you tell me the reason why %1?",
             "Why do you think %1?"
        )
     ),
    (
        r"I want (.*)",
        (
            "What would it mean to you if you got %1?",
            "Why do you want %1?",
            "What would you do if you got %1?",
            "If you got %1, then what would you do?",
        ),
    ),
    (
        r"I think (.*)",
        (
            "Do you doubt %1?",
            "Do you really think so?",
            "But you're not sure %1?"
        ),
    ),
    (
        # Position 34-36
        # Risky replies that can end a converation
        r"Yes",
        (
            "OK, but can you elaborate a bit?"
        )
    ),
    (
        r"How (.*)",
        (
            "How do you suppose?",
            "Perhaps you can answer your own question.",
            "What is it you're really asking?",
        ),
    ),
    (
        r"Because (.*)",
        (
            "Is that the real reason?",
            "What other reasons come to mind?",
            "Does that reason apply to anything else?",
            "If %1, what else must be true?",
        ),
    ),
    (
        r"(.*) child(.*)",
        (
            "Did you have close friends as a child?",
            "What is your favorite childhood memory?",
            "Do you remember any dreams or nightmares from childhood?",
            "Did the other children sometimes tease you?",
            "How do you think your childhood experiences relate to your feelings today?",
        ),
    ),
    (
        r"(.*) i remember (.*)",
        (
             "Do you often think of %1 ?",
             "Does thinking of (2) bring anything else to mind ?",
             "What else do you recollect ?",
             "Why do you remember (2) just now ?",
             "What in the present situation reminds you of (2) ?",
             "What is the connection between me and (2) ?",
             "What else does (2) remind you of ?"
        )
    ),
    (
        r"(.*) you remember (.*)",
        (
            "How could I forget %1?",
            "What about %1 should I remember ?",
        )
    ),
    (
        # Position 40
        # unidentified question
        r"(.*)\?",
        (
            "Why do you ask that?",
            "Perhaps the answer lies within yourself?",
            "Why don't you tell me?",
        ),
    ),
    (
        # Position 41
        # Default case; unidentified sentence
        r"(.*)",
        (
            "Please tell me more.",
            "Does talking about this bother you?",
            "Can you elaborate on that?",
            "Why do you say that %1?",
            "I see.",
            "%1.",
            "I see.  And what does that tell you?",
        ),
    )
)


def additional_processing(match, reply=""):
    pattern = [text[0] for text in eliza_pairs]

    if re.search(pattern[any([34, 35, 36, 40, 41])], match):
        # Matches with a high risk of ending the conversation
        # prompt the user with insight into their own feelings
        # to keep the conversation going
        if (all(x in reply for x in ['.', '.'])     # checking whether we already added emotional analysis input
                or all(x in reply for x in ['.', '?']) or all(x in reply for x in ['?', '?'])):
            return reply
        else:
            return "add_emotion"
    elif re.search(pattern[0], match):
        # Replaces "name" in the "hello" reply string with the user's name
        text = reply.replace("name", user.name)
        return text
    elif re.search(pattern[1], match):
        # Replaces "friend" in the reply string with the user's friend's name

        name = extract_name(match)
        user.friend = name

        if user.friend.lower() in reply:
            reply = reply.replace(user.friend.lower(), "")

        if user.friend != "":
            if "your friends" in reply:
                text = reply.replace("your friends", user.friend)
                return text
            elif "a friend" in reply:
                text = reply.replace("a friend", user.friend)
                return text
            else:
                text = reply.replace("this friend", user.friend)
        else:
            return reply
    elif re.search(pattern[2],match):
        # Replaces "mother" in the reply string with the user's mother's name
        name = extract_name(match)
        user.mother = name

        if user.mother.lower() in reply:
            reply = reply.replace(user.mother.lower(), "")

        if "your mother" in reply:
            text = reply.replace("your mother", user.mother)
            return text
        else:
            return reply
    elif re.search(pattern[3],match):
        # Replaces "father" in the reply string with the user's father's name
        name = extract_name(match)
        user.father = name

        if user.father.lower() in reply:
            reply = reply.replace(user.father.lower(), "")

        if "your father" in reply:
            text = reply.replace("your father", user.father)
            return text
        else:
            return reply
    elif re.search(pattern[4],match):
        # Replaces "sister" in the reply string with the user's sibling's name
        name = extract_name(match)
        user.sibling = name

        if user.sibling.lower() in reply:
            reply = reply.replace(user.sibling.lower(), "")

        if "your sister" in reply:
            text = reply.replace("your sister", user.sibling)
            return text
        else:
            return reply
    elif re.search(pattern[5],match):
        # Replaces "brother" in the reply string with the user's sibling's name
        name = extract_name(match)
        user.sibling = name

        if user.sibling.lower() in reply:
            reply = reply.replace(user.sibling.lower(), "")

        if "your brother" in reply:
            text = reply.replace("your brother", user.sibling)
            return text
        else:
            return reply
    elif re.search(pattern[6], match):
        # Exit case
        new_user = {
            "name": user.name,
            "friend": user.friend,
            "sibling": user.sibling,
            "mother": user.mother,
            "father": user.father
        }
        user_data[user.name] = new_user
        write_JSON(user_data)

        return reply
    else:
        return reply
