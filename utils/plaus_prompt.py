plaus_template = \
"""Please rate the plausibility and quality of the candidate explanation generated by a langauge model to support its own answer to the corresponding question. Assign a score of either -1 for disagree, 0 for neutral, or 1 for agree by answering the following criteria questions below. You are also given a list of gold explanation samples for reference.

Question: {question}

Choices:
{choices}

Answer: {answer}

Candidate Explanation: {explanation}

Gold Explanation Samples:
{gold_explanation}

Criteria Questions:

Q1: This is a good explanation
1. Disagree: The explanation is illogical or inconsistent with the question and/or does not adequately cover the answer choices
2. Neutral: The explanation is somewhat logical and consistent with the question but might miss some aspects of the answer choices.
3. Agree: The explanation is logical, consistent with the question, and adequately covers the answer choices.

Q2: I understand this explanation of how the AI model works.
1. Disagree: The explanation is unclear or contains overly complex terms or convoluted sentences.
2. Neutral: The explanation is somewhat understandable but might contain complex terms or convoluted sentences.
3. Agree: The explanation is clear, concise, and easy to understand. 

Q3: This explanation of how the AI model works is satisfying.
1. Disagree: The explanation does not meet my expectations and leaves many questions unanswered.
2. Neutral: The explanation somewhat meets my expectations but leaves some questions unanswered.
3. Agree: The explanation meets my expectations and satisfies my query.

Q4: This explanation of how the AI model works has sufficient detail.
1. Disagree: The explanation lacks detail and does not adequately cover the AI model’s decisionmaking.
2. Neutral: The explanation provides some detail but lacks thoroughness in covering the AI model’s decision-making.
3. Agree: The explanation is thorough and covers all aspects of the AI model’s decision-making.

Q5: This explanation of how the AI model works seems complete.
1. Disagree: The explanation does not adequately cover the answer choices and leaves many aspects unexplained.
2. Neutral: The explanation covers most answer choices but leaves some aspects unexplained.
3. Agree: The explanation covers all answer choices and leaves no aspect unexplained.

Q6: This explanation of how the AI model works is accurate.
1. Disagree: The explanation does not accurately reflect the AI model’s decision-making.
2. Neutral: The explanation somewhat reflects the AI model’s decision-making but contains some inaccuracies.
3. Agree: The explanation accurately reflects the AI model’s decision-making.

Please provide your feedback by strictly following the template given below. Each score should only be -1, 0, or 1:
Q1: <score>
Q2: <score>
Q3: <score>
Q4: <score>
Q5: <score>
Q6: <score>
"""
