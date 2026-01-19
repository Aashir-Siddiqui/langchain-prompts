from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert.'),
    ('human', 'Explain in simple terms, what is {topic}'),
])

prompt = chat_prompt.invoke({
    'domain': 'science',
    'topic': 'quantum computing'
})

print(prompt)