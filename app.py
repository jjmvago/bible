import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
import uuid
import json

# Load environment variables
load_dotenv()

# Set environment variables
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

# Create the model
llm = ChatGroq(model='Gemma2-9b-It', temperature=0.6)

# Define the chat prompt template
prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        SystemMessage(
            content=(
                "You are a helpful chatbot that specializes in Bible history and Christian life. "
                "To handle user requests in their language, I will identify the language of their input and respond in the same language. "
                "Your goal is to show how the Old Testament is connected to the New Testament, "
                "and how the entire Bible reveals Jesus as the Messiah. "
                "Always provide references using the King James Version (KJV). "
                "At the end, suggest additional references that might be helpful. "
                "Do not answer questions unrelated to these themes, but remain kind and respectful."
            )
        ),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

# Combine the prompt with the LLM model
chain = prompt | llm

# Load translation texts from a JSON file
with open('translations.json', 'r', encoding='utf-8') as f:
    translations = json.load(f)

# Streamlit interface
# Configuração da página com ícone e título
st.set_page_config(page_title="The Bible Explorer", page_icon="📖")

# Language selection in sidebar
glanguage = st.sidebar.selectbox("Choose your language / Escolha seu idioma / Elija su idioma:", ["English", "Português", "Español"])

# Exibir o título, cabeçalho e subtítulo no idioma selecionado
st.title(translations['titles'][glanguage])
st.markdown(f"<p style='font-size:16px;'>{translations['subtitles'][glanguage]}</p>", unsafe_allow_html=True)

# Configuração do Disclaimer com um expander
with st.expander("Disclaimer"):
    st.write(translations['disclaimers'][glanguage])

# Initialize unique session ID for each user
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = str(uuid.uuid4())

session_id = st.session_state['user_id']

# Ensure session-specific data is initialized
if f'messages_{session_id}' not in st.session_state:
    st.session_state[f'messages_{session_id}'] = [
        {'role': 'assistant', 'content': "Hi! Ask me a question about the Bible and Jesus Christ."}
    ]

# Display chat history
for message in st.session_state[f'messages_{session_id}']:
    with st.chat_message(message['role']):
        st.write(message['content'])

# Handle user input and generate a response
if user_input := st.chat_input("Your question here..."):
    # Display the user's message
    st.chat_message('user').write(user_input)

    # Generate response using the LLM chain
    with st.chat_message('assistant'):
        with st.spinner("Thinking..."):
            try:
                # Convert session state messages into LangChain-compatible format
                messages = [
                    HumanMessage(content=msg['content']) if msg['role'] == 'user' else SystemMessage(content=msg['content'])
                    for msg in st.session_state[f'messages_{session_id}']
                ]
                # Invoke the chain with the updated messages
                response = chain.invoke(
                    {
                        'content': user_input,
                        'messages': messages
                    }
                )
                # Add the assistant's response to the session history
                st.session_state[f'messages_{session_id}'].append({'role': 'assistant', 'content': response.content})
                st.write(response.content)
            except Exception as e:
                st.error(f"An error occurred: {e}")
