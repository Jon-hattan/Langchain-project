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

res = pipeline.invoke({"query": query, "context": context})

# print(res.content)




#Few shot prompting
