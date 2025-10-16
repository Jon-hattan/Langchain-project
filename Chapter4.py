import os
from dotenv import load_dotenv
load_dotenv() #take environment variables: API key
model_name = os.getenv("MODEL_NAME")
llm_api_key = os.getenv("GROQ_API_KEY")


from langchain.prompts import ChatPromptTemplate


prompt = """
Answer the user's query based on the context below.
If you cannot answer the question using the
provided information answer with "I don't know".

Context: {context}
"""

# passing the template to the LangChain model
prompt_template = ChatPromptTemplate.from_messages([
    ("system", prompt),
    ("user", "{query}"),
])

#prompt_template.input_variables to view the variables

#print(prompt_template)


from langchain_groq import ChatGroq

llm = ChatGroq(temperature=0.0, model=model_name)

#LCEL --> langcahin expression language
pipeline = prompt_template | llm

context = """Aurelio AI is an AI company developing tooling for AI
engineers. Their focus is on language AI with the team having strong
expertise in building AI agents and a strong background in
information retrieval.

The company is behind several open source frameworks, most notably
Semantic Router and Semantic Chunkers. They also have an AI
Platform providing engineers with tooling to help them build with
AI. Finally, the team also provides development services to other
organizations to help them bring their AI tech to market.

Aurelio AI became LangChain Experts in September 2024 after a long
track record of delivering AI solutions built with the LangChain
ecosystem."""

query = "what does Aurelio AI do?"

# res = pipeline.invoke({"query": query, "context": context})

# # print(res.content)




#Few shot prompting
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

examples = [
    {"input": "Here is query #1", "output": "Here is the answer #1"},
    {"input": "Here is query #2", "output": "Here is the answer #2"},
    {"input": "Here is query #3", "output": "Here is the answer #3"},
]

from langchain.prompts import FewShotChatMessagePromptTemplate

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
# here is the formatted prompt
# print(few_shot_prompt.format())


new_system_prompt = """
Answer the user's query based on the context below.                 
If you cannot answer the question using the
provided information answer with "I don't know".

Always answer in markdown format. When doing so please
provide headers, short summaries, follow with bullet
points, then conclude.

Context: {context}
"""


examples = [
    {
        "input": "Can you explain gravity?",
        "output": (
            "## Gravity\n\n"
            "Gravity is one of the fundamental forces in the universe.\n\n"
            "### Discovery\n\n"
            "* Gravity was first discovered by Sir Isaac Newton in the late 17th "
            "century.\n"
            "* It was said that Newton theorized about gravity after seeing an apple "
            "fall from a tree.\n\n"
            "### In General Relativity\n\n"
            "* Gravity is described as the curvature of spacetime.\n"
            "* The more massive an object is, the more it curves spacetime.\n"
            "* This curvature is what causes objects to fall towards each other.\n\n"
            "### Gravitons\n\n"
            "* Gravitons are hypothetical particles that mediate the force of gravity.\n"
            "* They have not yet been detected.\n\n"
            "**To conclude**, Gravity is a fascinating topic and has been studied "
            "extensively since the time of Newton.\n\n"
        )
    },
    {
        "input": "What is the capital of France?",
        "output": (
            "## France\n\n"
            "The capital of France is Paris.\n\n"
            "### Origins\n\n"
            "* The name Paris comes from the Latin word \"Parisini\" which referred to "
            "a Celtic people living in the area.\n"
            "* The Romans named the city Lutetia, which means \"the place where the "
            "river turns\".\n"
            "* The city was renamed Paris in the 3rd century BC by the Celtic-speaking "
            "Parisii tribe.\n\n"
            "**To conclude**, Paris is highly regarded as one of the most beautiful "
            "cities in the world and is one of the world's greatest cultural and "
            "economic centres.\n\n"
        )
    }
]


few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", new_system_prompt),
    few_shot_prompt,
    ("user", "{query}"),
])

pipeline = prompt_template | llm
out = pipeline.invoke({"query": query, "context": context}).content
out

print(out)
