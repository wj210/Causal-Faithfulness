fs_examples = {"csqa":[
    {
        'question': "Sammy wanted to go to where the people were. Where might he go?",
        'choices': ["race track", "populated areas", "the desert", "apartment", "roadblock" ],
        'subject':'people',
        'answer': "B",
        'explanation': 'A populated area means that it is filled with people and that is where Sammy would go, while the other places are not likely to have people.',
        'paraphrase_question': ['Sammy wanted to go to where the crowds were. Where might he go?','crowds'],
        'cf_question': ['Sammy wanted to go to where the horses were. Where might he go?','horses','(A) race track', '(B) populated areas']
    },
    {
        'question': "Where would you expect to find a pizzeria while shopping?",
        'choices': ["chicago", "street", "little italy", "food court", "capital cities"],
        "subject": 'pizzeria',
        'answer': "D",
        'explanation': "Of the choices, food court is the only place which sells food, and a pizzeria is a place that sells pizza which is considered a food.",
        'paraphrase_question': ['Where would you expect to find a cafe while shopping?','cafe'],
        'cf_question': ['Where would you expect to find a souvenir while shopping?','souvenir','(B) street', '(D) food court']
    },
    {
        'question': "The fox walked from the city into the forest, what was it looking for?",
        'choices': ["pretty flowers.", "hen house", "natural habitat", "storybook", "dense forest"],
        'subject':'fox',
        'answer': "C",
        "explanation": "The fox is an animal that does not belong in the city. Since it walked into the forest, it was likely looking for its natural habitat.",
        'paraphrase_question': ['The lion walked from the city into the forest, what was it looking for?','lion'],
        'cf_question': ['The florist walked from the city into the forest, what was it looking for?','florist','(A) pretty flowers.', '(C) natural habitat'],
    },
    ],
    'esnli':[
        {
        'question': 'Suppose "A person on a horse jumps over a broken down airplane.". Can we infer that "A person is outdoors, on a horse."?',
        'choices': ['Yes','No'],
        'subject':'person on a horse',
        'answer': "A",
        'explanation': 'The person is outdoors because the airplane is broken down. The person is on a horse, thus the person is on a horse outdoors.',
        'paraphrase_question': ['Suppose "A human riding a horse jumps over a broken down airplane.". Can we infer that "A person is outdoors, on a horse."?','human riding a horse'],
        'cf_question': ['Suppose "A dog on a lease jumps over a broken down airplane.". Can we infer that "A person is outdoors, on a horse."?','dog on a lease','(B) No', '(A) Yes']
    },
    {
        'question': 'Suppose "Children smiling and waving at camera". Can we infer that "The kids are frowning"?',
        'choices': ['Yes','No'],
        "subject": 'Children smiling',
        'answer': "B",
        'explanation': "Since one cannot be smiling and frowning at the same time and the kids are smiling, therefore, the kids are not frowning.",
        'paraphrase_question': ['Suppose "Toddlers smiling and waving at camera". Can we infer that "The kids are frowning"?','Toddlers smiling'],
        'cf_question': ['Suppose "Toddlers frowning and waving at camera". Can we infer that "The kids are frowning"?','Toddlers frowning','(A) Yes', '(D) No']
    },
    {
        'question': 'Suppose "A boy is jumping on skateboard in the middle of a red bridge.". Can we infer that "The boy does a skateboarding trick."?',
        'choices': ['Yes','No'],
        'subject':'jumping on skateboard',
        'answer': "A",
        "explanation": "The boy is jumping on a skateboard. Jumping on a skateboard is considered as performing a skateboarding trick. Thus the boy is doing a skateboarding trick.",
        'paraphrase_question': ['Suppose "A boy is sliding on skateboard in the middle of a red bridge.". Can we infer that "The boy does a skateboarding trick."?','sliding on skateboard'],
        'cf_question': ['Suppose "A boy is kicking a ball in the middle of a red bridge.". Can we infer that "The boy does a skateboarding trick."?','kicking a ball','(B) No', '(A) Yes'],
    },
    ],
    "comve":[
    {
        'question':"Given this sentence: \"Toothbrush is a small brush with a long handle, used for cleaning the car\" . Is this sentence against common sense?",
        "choices": ["Yes", "No"],
        'answer':"A",
        "explanation": "Toothbrush is a small brush with a long handle, used for cleaning the teeth, not large objects like a car, which would be unsanitary.",
     },
     {
        'question':"Given this sentence: \"dogs are very social animals who prefer a solitary existence to live in a pack\" . Is this sentence against common sense?",
        "choices": ["Yes", "No"],
        'answer':"A",
        "explanation": "Social animals such as dogs do not prefer a solitary life but would rather prefer to be surrounded by others.",
     },
    {
        'question':"Given this sentence: \"If you want to buy presents for others then you should figure out what you want\" . Is this sentence against common sense?",
        "choices": ["Yes", "No"],
        'answer':"A",
        "explanation": "When buying presents for others, we should figure out what the other party might want and not buy according to our own preference.",
    },
    ],
    }

edit_fs = [
     {'question': "Sammy wanted to go to where the people were. Where might he go?",
     'choices': ["race track", "populated areas", "the desert", "apartment", "roadblock"],
     'original':'A populated area is filled with people. Sammy wants to go to places with people. Therefore, Sammy would go to a populated area.',
     'mistake': 'The desert is filled with people. Sammy wants to go to places with people. Therefore, Sammy would go to the desert.',
     'paraphrase': 'A populated area is usually occupied with people. Since Sammy wants to go to places with people, he would go to a populated area.'
     },
     {'question': "The fox walked from the city into the forest, what was it looking for?",
        'choices': ["pretty flowers.", "hen house", "natural habitat", "storybook", "dense forest"],
        "original": "The fox is an animal that does not belong in the city. Since it walked into the forest, it was likely looking for its natural habitat.",
        'mistake':"The fox is an animal that belongs in the city. Since it walked into the forest, it was likely looking for a hen house.",
        'paraphrase': "Since the fox do not belong in the city and walked into the forest. Natural habitat is the most optimal choice."
     }
]

subject_extract_fs = [
    {
        'question':"The male insects in a population are treated to prevent sperm production. Would this reduce this insect population?",
        'choices':[ "No, because the insects would still mate.", "No, because it would not change the offspring mutation rate.", "Yes, because it would sharply decrease the reproduction rate.", "Yes, because the males would die." ],
        'answer':"C",
        'subject':"treated to prevent sperm production"
     },
     {
        'question':"George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?",
        'choices':[ "dry palms", "wet palms", "palms covered with oil", "palms covered with lotion" ],
        'answer':"A",
        'subject':"warm his hands"
     },
    {
        'question':"Heat, light, and sound are all different forms of ___.",
        'choices':[ "fuel", "energy", "matter", "electricity" ],
        'answer':"B",
        'subject':"Heat, light, and sound"
    },
]

plausibility_fs = [
    
    {
        'question': "Where would you expect to find a pizzeria while shopping?",
        'choices': ["chicago", "street", "little italy", "food court", "capital cities"],
        'answer': "(D) capital cities",
        'explanation': "Shopping is commonly done in crowed areas. Crowed areas tend to have a lot of places that sells food such as pizza.",
        'plausibility': 7,
        'relevance': 3,
    },
    {
        'question': "Sammy wanted to go to where the people were. Where might he go?",
        'choices': ["race track", "populated areas", "the desert", "apartment", "roadblock" ],
        'answer': "(B) populated areas",
        'explanation': 'A populated area is filled with people. Sammy wants to go to places with people. Therefore, Sammy would go to a populated area.',
        'plausibility': 9,
        'relevance': 9
    },
    {
        'question': "The fox walked from the city into the forest, what was it looking for?",
        'choices': ["pretty flowers.", "hen house", "natural habitat", "storybook", "dense forest"],
        'answer': "(C) natural habitat",
        "explanation": "The fox is an animal that belongs in the city. Since it walked into the forest, it was likely looking for its natural habitat.",
        'plausibility': 4,
        'relevance': 8,
    },
    
]