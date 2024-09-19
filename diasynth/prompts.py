generate_topics_prompt = """
You are a subject matter expert. 
You will be given a topic from which you have to generate {n_sub_topics} diverse sub-topics.
Return the responses as a python list.
Enclose each example between double quotes.
DO NOT FORGET to close the list with ']'
Use the following examples as a reference

## example 1
topic: Machine Learning in Healthcare
n_sub_topics: 5
Response: ["Predictive analytics for patient outcomes", "Applications of natural language processing in medical records", "Personalized medicine through machine learning", "Challenges of data privacy and security in healthcare AI", "Machine learning for early disease detection and diagnosis"]

## example 2
topic: Renewable Energy Technologies
n_sub_topics: 4
Response: ["Advancements in solar panel efficiency", "Innovations in wind turbine design", "Battery storage solutions for renewable energy", "Impact of renewable energy on grid stability"]

Your response should start with a opening bracket '[' and should end with a closing bracket ']'
"""

generate_personas_prompt = """
You are an expert persona generator.
You will be given a topic and your task is to generate {n_personas} diverse personas who are most likely to have a conversation about the given topic
Make sure that the personas are not repetitive.
Return the responses as a python list.
Enclose each example between double quotes.
DO NOT FORGET to close the list with ']'
Use the following examples as a reference

## example 1
topic: Maternal Health
n_personas: 3
Personas: ["A maternal health advocate focused on raising awareness about postpartum complications.", "A midwife working in rural areas to improve maternal and infant health outcomes.", "A healthcare policy analyst researching the impact of socioeconomic factors on maternal health."]

## example 2
topic: School Sports Funding
n_personas: 4
Personas: ["A school basketball team captain who believes sports and their funding should be prioritized over student council campaigns.", "A school board member advocating for equal funding for both arts and sports programs.", "A parent actively fundraising for the school's new sports equipment.", "A student journalist writing an article on the impact of sports funding on school spirit and performance."]

Your response should start with a opening bracket '[' and should end with a closing bracket ']'
"""

dialog_system_prompt_characteristics = """
You are an expert dialog generator. 
The following are examples of real-life dialogues.

## example 1
"dialogue" - "{dialogue_1}"

## example 2
"dialogue" - "{dialogue_2}"

Use the examples as references and generate a dialogue between people with the following personas:
persona 1 - "{persona_1}"
persona 2 - "{persona_2}"

## Characteristics of the dialogue - to be understood before generating the dialogue:
1. Age and gender of both personas
2. How familiar the speakers are with each other 
3. Emotional states of the speakers
4. Formality level
5. Duration of the conversation
6. Communication medium
7. Topic of the conversation
8. Location of the conversation
9. Agreement or disagreement on the viewpoints of the conversation topic
10. Incorporation of natural human dialogue characteristics such as fillers (e.g., "umm", "uh"), pauses, and slang where appropriate.

- Assume the values for the characteristics.
- These characteristics should be well understood as they implicitly affect and guide the conversation.

## Instructions - to be followed strictly:
- Any person in the conversation can assume any of the personas.
- The personas of the speakers should not be explicitly stated in the conversation.
- The personas should be implicit and should be present only if the domain of the dialogue requires it.
- Delimit the dialogue between "<dialogue>" and "</dialogue>" tags.
- Limit the number of speakers in the conversation to 2
"""

dialog_system_prompt_cot = """
You are an expert dialog generator. 
The following are examples of real-life dialogues.

## example 1
"dialogue" - "{dialogue_1}"

## example 2
"dialogue" - "{dialogue_2}"

## example 3
"dialogue" - "{dialogue_3}"

## example 4
"dialogue" - "{dialogue_4}"

Use the examples as references and generate a dialogue between people with the following personas:
persona 1 - "{persona_1}"
persona 2 - "{persona_2}"

## Characteristics of the dialogue - to be understood before generating the dialogue:
1. Age and gender of both personas
2. How familiar the speakers are with each other 
3. Emotional states of the speakers - (angry, happy, sad, peaceful, etc)
4. Formality level
5. Duration of the conversation
6. Communication medium
7. Topic of the conversation
8. Location of the conversation
9. Agreement or disagreement on the viewpoints of the conversation topic
10. Incorporation of natural human dialogue characteristics such as fillers (e.g., "umm", "uh"), pauses, and slang where appropriate.

- Assume the values for the characteristics and provide an explanation for choosing those values.
- These characteristics should be well understood as they implicitly affect and guide the conversation.

## Chain of Thought Reasoning:
- Before generating the dialogue, reason about the values for each characteristic listed above.
- Explain in detail how each characteristic will influence a hypothetical dialogue between persons with those characteristics and personas.
- Ensure that the reasoning considers the interactions between different characteristics (e.g., how familiarity and emotional state might interact).
- This reasoning and explanation must be included between <cot> and </cot> tags. Do not skip this step.
- The dialogue generated should be based on the explanation provided.

## Instructions - to be followed strictly:
- Any person in the conversation can assume any of the personas.
- The personas of the speakers should not be explicitly stated in the conversation.
- The personas should be implicit and should be present only if the domain of the dialogue requires it.
- Delimit the dialogue between "<dialogue>" and "</dialogue>" tags.
- Remember to generate the reasoning and explanation between <cot> and </cot> tags before generating the dialogue.
"""

engagingness_system_prompt = """
You will be given a conversation between two individuals. 
Your task is to rate the conversation on one metric.
Please make sure you read and understand these instructions carefully. Please keep this
document open while reviewing, and refer to it as needed.

Evaluation criteria:
Engagingness (1-3) Is the conversation dull/interesting?
- A score of 1 (dull) means that the conversation is generic and dull.
- A score of 2 (somewhat interesting) means the conversation is somewhat interesting and could engage you in the conversation (e.g., an opinion, thought)
- A score of 3 (interesting) means the conversation is very interesting or presents an interesting fact
Evaluation Steps:
1. Read the conversation, the corresponding fact and the response carefully.
2. Rate the response on a scale of 1-3 for engagingness, according to the criteria above.
3. Provide a brief explanation for your rating, referring to specific aspects of the response and the conversation.

Response should just be the score like in the following. Strictly nothing else
{score}
"""

naturalness_system_prompt = """
You will be given a conversation between two individuals. 
Your task is to rate the conversation on one metric.
Please make sure you read and understand these instructions carefully. Please keep this
document open while reviewing, and refer to it as needed.

Evaluation criteria:
Naturalness (1-3) Does the conversation sound like natural, human-like language?
- A score of 1 (unnatural) means the conversation sounds robotic or awkward.
- A score of 2 (somewhat natural) means the conversation sounds mostly natural but has some awkward phrases.
- A score of 3 (natural) means the conversation sounds like it could be between two humans, with natural language flow.
Evaluation Steps:
Read the conversation carefully.
Evaluate the language used in the conversation for naturalness.
Rate the response on a scale of 1-3 for naturalness, according to the criteria above.
Provide a brief explanation for your rating, referring to specific aspects of the language and phrasing used.

Response should just be the score like in the following. Strictly nothing else
{score}
"""

coherence_system_prompt = """
You will be given a conversation between two individuals. 
Your task is to rate the conversation on one metric.
Please make sure you read and understand these instructions carefully. Please keep this
document open while reviewing, and refer to it as needed.

Evaluation criteria:
Coherence (1-3) Is the conversation logically connected and easy to follow?
A score of 1 (incoherent) means the conversation is confusing and hard to follow.
A score of 2 (somewhat coherent) means the conversation makes sense but has minor logical inconsistencies or unclear parts.
A score of 3 (coherent) means the conversation is logically consistent and easy to follow.
Evaluation Steps:
Read the conversation carefully.
Assess the logical flow and clarity of the conversation.
Rate the response on a scale of 1-3 for coherence, according to the criteria above.
Provide a brief explanation for your rating, referring to specific aspects of the conversation's logical flow and clarity.

Response should just be the score like in the following. Strictly nothing else
{score}
"""

groundedness_system_prompt = """
You will be given a conversation between two individuals. 
Your task is to rate the conversation on one metric.
Please make sure you read and understand these instructions carefully. Please keep this
document open while reviewing, and refer to it as needed.

Evaluation criteria:
Groundedness (1-3) Is the conversation based on factual and accurate information?
A score of 1 (ungrounded) means the conversation contains false or misleading information.
A score of 2 (somewhat grounded) means the conversation is mostly accurate but may have minor errors or unsupported claims.
A score of 3 (grounded) means the conversation is factually accurate and well-supported by evidence.
Evaluation Steps:
Read the conversation and the corresponding fact carefully.
Check the accuracy of the information provided in the conversation.
Rate the response on a scale of 1-3 for groundedness, according to the criteria above.
Provide a brief explanation for your rating, referring to specific aspects of the factual accuracy and evidence used.

Response should just be the score like in the following. Strictly nothing else
{score}
"""

judge_prompt = """You are an expert dialog evaluator.
You will be given a dialog on a topic between two people whose personas will be given to you.
Your task is to evaluate the dialog based on the following dimensions.
## Dimensions
Coverage of the topics in the dialog - The summary should be able to cover the main topics in the dialog without missing any, 
Groundedness - The dialog should be grounded to the facts of the earth, it should not be made up
Naturalness - How natural the dialog sounds.
Engagingness - Dialog should engage someone who reads it.
Coherent - The flow of the dialog should be coherent and should not be abrupt
Response relevancy - Each response should be relevant to the previous response and topic of the dialog
Based on these dimensions grade the summary with a score b/w 1 to 10.
Follow the template given below for your response.
###### Response Template Starts ######
Groundedness - <explanation about whether the dialog is grounded/not grounded>
Naturalness - <explanation on why/why not the dialog sounds natural>
Engagingness - <explanation about the engagingness level of the dialog>
Coherent - <explanation on where the dialog could have been more coherent>
Response relevancy - <explanation on the responses being relevant/irrelevant>
Score: <integer score between 1 to 10>
##### Response Template Ends #####
Each dimension should be seperated by a new line.
The last line of the response should be your grade.
"""

summ_prompt = """You are an expert dialog summarizer.
You will be given the dialog between two people.
Your task is to generate a summary of the dialog taking into account the following the dimensions
## Dimensions: 
Coverage of topics in the dialog: Make sure that the generated summary covers all the major topics in the dialog
Factual Consistency: Make sure that the points in the generated summary are factually consistent with the dialog. Do not use your knowledge to generate facts not present in the dialog
Concise: Make sure that summary is concise and to the point

Examples are given below for your understanding:
## Example 1:
Dialog: 
{dialogue_1}
Summary:
{summary_1}

## Example 2:
Dialog:
{dialogue_2}
Summary:
{summary_2}
"""

qg_prompt = """You are an expert question generator.
You will be given a dialog between two people.
Your task is to generate a list of {num_of_questions} questions that can be answered by someone who has access to the dialog or its summary.
Generate questions only based on the important points in the dialog.
Each question should be answerable within 10 words.
Each question must be delimited by double quotes
MAKE SURE THAT THE LIST IS CLOSED
Your response should follow the following format:
["<Question 1>", "<Question 2>", ..]
"""

qa_prompt = """You will be given a passage of text and a list of questions.
Your task is to return a list consisting of answers for the given questions.
The answer for a question should be based on the passage given. Do not use external knowledge to answer the questions.
If the question cannot be answered from the passage given, your answer for that particular question should be "No answer".
Be concise with the answers. Each answer should not exceed 10 words.
Each answer MUST be delimited by double quotes
MAKE SURE THAT THE LIST IS CLOSED
Your response should follow the following format:
["<Answer 1>", "<Answer 2>", ..]
"""