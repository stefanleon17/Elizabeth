# Elizabeth
A Python command line chatbot that combines ELIZA with BERT-based emotion analysis and NER


Elizabeth: A Rogerian Therapy Chatbot Combining ELIZA and BERT

Introduction
ELIZA, as originally envisioned by Joseph Weizenbaum (1966) is a chatbot that functions according to a keyword-based pattern matching algorithm, and creates conversations by reflecting back the words of the user, in a way roughly similar to Rogerian psychotherapists – this choice was essential, as Rogerian psychotherapy usually requires one to assume a role of knowing nothing about the patient, and to ask questions that invite them to think about solutions to their problems on their own. While it was initially meant to underline the shallowness of human-computer conversations, ELIZA ended up incredibly successful at convincing people that it was a complex artificial intelligence, close to the likes of the 2001: A Space Odyssey’s (1968) supercomputer HAL (Wardrip-Fruin, 2007). This led researchers to conclude that people have a strong tendency to anthropomorphize and project a high level of intelligence on computer programs capable of processing and recreating human language – this phenomenon was appropriately dubbed the “ELIZA effect” (Ekbia, 2008). 
However, despite its unexpected levels of efficacy, there are still several big shortcomings of the original program: it fails to maintain conversations, it has no reasoning or learning mechanisms, and it fails to consider the context of the conversation – beyond the context conveyed through its inbuilt pattern-matching system (Xu & Zhuang, 2020). There have been more recent attempts at making psychotherapy chatbots, implementing different therapeutic frameworks, aiming for therapeutic effectiveness, while trying to take advantage of the best language models available: most notably, Woebot (https://woebothealth.com/) is a well-researched and fairly effective chatbot that utilizes techniques from Cognitive Behavioral Therapy, Interpersonal Psychotherapy and Dialectical Behavior Therapy, alongside modern Natural Language Processing (NLP) to provide users with an alternative to in-person therapy. The implications of such a product are wide-spanning and complex, but the major benefit is apparent: it eliminates many of the initial barriers to seeking help via in-person therapy, like scheduling, limited availability, and the negative social feedback associated with seeking mental health services (Vanheusden et al., 2008).
However, it seems like the original ELIZA design has been mostly abandoned, despite promising initial results. This project aims to improve upon this design, by using modern NLP techniques to incorporate emotional analysis and named entity recognition into an ELIZA-inspired chatbot, in order to better maintain conversations, and consequentially provide a better conversational experience for its users. The result is named ELIZABETH, paying homage to the original ELIZA, and to the BERT-based transformer models it makes use of.
Methods
The core of the program is an ELIZA-inspired pattern-matching algorithm. This was implemented through the NLTK python library, by making use of their pre-defined reflections dictionary – which allows the chatbot to use the correct pronouns when reflecting back the user’s words, e.g. converting “I am sad” to “you are sad” – and the respond method – which performs the pattern-matching required to understand the user’s question and respond accordingly – defined in the Chat class for chatbots. For the definition of recognizable patterns and proposed responses, it uses a modified version of the list defined within the nltk.chat.eliza library. Additional cases were added, to cover a wider range of replies, in an attempt to provide a more varied response; furthermore, certain pre-determined replies were modified or eliminated, on grounds of not prompting a clear enough response from users in preliminary testing of the algorithm. Additional processing was also employed, modifying non-specific references to the user and its friends and family, with specific names, where available. Pattern-matched replies were also improved upon with the addition of emotion-based replies: after processing and matching the user’s textual input, ELIZABETH periodically adds an additional sentence, reflecting the program’s understanding of their emotional state – e.g., “Do you feel angry?” The purpose of this feature is prompting the user to think about their own emotional state, and to their raise level of engagement.
For implementing emotion analysis, ELIZABETH uses the Hugging Face BERT-based transformer available at bdotloh/distilbert-base-uncased-empathetic-dialogues-context (https://huggingface.co/bdotloh/distilbert-base-uncased-empathetic-dialogues-context). This transformer classifies the input sentence into one of 32 emotional categories, and is used to provide insight into the user’s emotional state. Prior to using this transformer, ELIZABETH used the transformer at finiteautomata/bertweet-base-emotion-analysis (https://huggingface.co/finiteautomata/bertweet-base-emotion-analysis). This transformer only classifies sentences into one of 7 emotional categories (based on the 6 commonly accepted basic human emotions, and an “others” category); this was rejected in favor of the newer model, because the emotional input it provided ended up being too repetitive to promote user engagement.
For implementing named entity recognition, ELIZABETH made use of the basic BERT-based model available at dslim/bert-base-NER (https://huggingface.co/dslim/bert-base-NER). This model extracts names of persons, organizations and locations (with an additional MISC category, for names that do not fit into either); the program used this transformer in order to learn the user’s name, as well as the names of their parents, a friend, and one of their siblings, if provided at any point in the conversation. These names are stores as strings in a separate JSON file.
In order to evaluate results, I asked several other college students (n=4) to interact with the chatbot twice: first, with the basic ELIZA model provided in the nltk.chat.eliza library, and second, with the ELIZABETH model (that also includes emotion analysis and named entity recognition). Their performance indicators tracked were their time spent engaging with the chatbot (Engagement Time), as well as their feelings towards whether they felt the interaction was improved by these features (Interaction Quality). Additionally, participants were queried on their perceived degree of Conversational Coherence – i.e., whether they felt the chatbot attempted to switch topics or alter the flow of conversation. 
Results
The participants considered ELIZABETH an overall improvement over ELIZA. The average Engagement Time went from 3 minutes and 30 seconds to 4 minutes and 27 seconds; the participants seemed to entertain ELIZABETH’s questions about their emotional states, even pausing to think about it before replying. In a post-trial survey, 3 of the participants reported that the Interaction Quality was better with ELIZABETH; the fourth participant considered ELIZA and ELIZABETH to be fairly similar in their conversation quality. In terms of Conversational Coherence, ELIZA was seen as very incoherent, and all participants reported feeling annoyed at its lack of context awareness; ELIZABETH fell into similar pitfalls however, with users also reporting it struggled to follow the flow of conversation. All 4 participants rated ELIZA and ELIZABETH similarly in terms of Conversational Coherence.
Discussion
It seems like the addition of emotional analysis to the ELIZA model can address concerns related to sustaining conversations and maintaining engagement: users seemed significantly more invested in replying, and even arguing with ELIZABETH’s understanding of their emotions, compared to when talking to ELIZA. However, moderation is likely key: in preliminary testing, constantly querying the user with sentences like “Do you feel sad?” was highly annoying and detrimental to the interaction. Additionally, the contextless nature of the emotion analysis model would sometimes cause ELIZABETH to change its evaluation of the user’s emotional state radically within a single sentence, thus decreasing its degree of Conversation Coherence. A basic understanding of human mood, perhaps through a mathematical formula to represent “emotional inertia,” would likely benefit the chatbot. 
Named entity recognition did not come into play much, besides reading and writing the user’s name. Considering that following context was an important feature, according to the participants, this model might be better exploited within a context-understanding system, rather than just recognizing people and names.
While Rogerian psychotherapy has not been at the forefront of its field in recent years, it is still true that chatbots could be the perfect Rogerian therapists: they can provide conversation with absolutely no external direction, thus better allowing the users to tap into their inner world; it can provide conversation completely devoid of judgement, which is often difficult, even for trained professionals; finally, it can be a perfect machine for listening and reflection, something humans can reasonably struggle with (Kensit, 2000). Given this untapped niche, it would probably be worthwhile to reopen the exploration of these therapy methods, through the use of intelligently programmed chatbots; and in understanding how these chatbots could be programmed to create better interactive experiences, this paper provides a stepping stone, on the road from ELIZA to a comprehensive AI therapist.
 
References
Ekbia, H. R. (2008). Artificial Dreams. https://doi.org/10.1017/cbo9780511802126
Kensit, D. A. (2000). Rogerian theory: A critique of the effectiveness of pure client-centred therapy. Counselling Psychology Quarterly, 13(4), 345–351.
Vanheusden, K., Mulder, C. L., van der Ende, J., van Lenthe, F. J., Mackenbach, J. P., & Verhulst, F. C. (2008). Young adults face major barriers to seeking help from mental health services. Patient Education and Counseling, 73(1), 97–104. https://doi.org/10.1016/j.pec.2008.05.006
Wardrip-Fruin, N. (2007). Three Play Effects: Eliza, Tale-Spin, and SimCity. Digital Humanities. https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=48c8d01b39b536e3f03fb5fc6c1b5be0749f8687
Weizenbaum, J. (1966). ELIZA---a computer program for the study of natural language communication between man and machine. Communications of the ACM, 9(1), 36–45. https://doi.org/10.1145/365153.365168
Xu, B., & Zhuang, Z. (2020). Survey on psychotherapy chatbots. Concurrency and Computation: Practice and Experience. https://doi.org/10.1002/cpe.6170

