import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, trim_messages
import uuid
import json
from operator import itemgetter

# Load environment variables
load_dotenv()

# Set environment variables
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Create the model
llm = ChatGroq(model='Gemma2-9b-It')

# Define the chat prompt template
prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        SystemMessage(
            content=(
                "You are a helpful chatbot that specializes in Bible history and Christian life. "
                "To handle user requests in their language {language}, I will identify the language of their input and respond in the same language. "
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

# Function to add messages to session history
def add_message_to_history(session_id, role, content):
    """Adds a message to the session's message history."""
    if f'messages_{session_id}' not in st.session_state:
        st.session_state[f'messages_{session_id}'] = []
    st.session_state[f'messages_{session_id}'].append({'role': role, 'content': content})

# Trim messages to ensure token limit is respected
trimmer = trim_messages(
    max_tokens=1000,
    strategy="last",
    token_counter=llm,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

# Combine the prompt and LLM with the trimmer
chain = (
    RunnablePassthrough.assign(messages=itemgetter('messages'))
    | prompt
    | trimmer  # Trimmer acts only on input to the LLM
    | llm
)

# Initialize session
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = str(uuid.uuid4())

session_id = st.session_state['user_id']

# Load translation texts from a JSON file
with open('translations.json', 'r', encoding='utf-8') as f:
    translations = json.load(f)

# Streamlit interface
st.set_page_config(page_title="The Bible Explorer", page_icon="📖")

# Language selection in sidebar
glanguage = st.sidebar.selectbox("Choose your language / Escolha seu idioma / Elija su idioma:", ["English", "Português", "Español"])

# Display title, header, and subtitle in the selected language
st.title(translations['titles'][glanguage])
st.markdown(f"<p style='font-size:16px;'>{translations['subtitles'][glanguage]}</p>", unsafe_allow_html=True)

# Disclaimer with an expander
with st.expander("Disclaimer"):
    st.write(translations['disclaimers'][glanguage])

# Initialize session-specific data
if f'messages_{session_id}' not in st.session_state:
    st.session_state[f'messages_{session_id}'] = [
        {'role': 'assistant', 'content': "Hi! Ask me a question about the Bible and Jesus Christ."}
    ]

# Display chat history
for message in st.session_state[f'messages_{session_id}']:
    with st.chat_message(message['role']):
        st.write(message['content'])

# Handle user input
if user_input := st.chat_input("Your question here..."):
    # Add user message to history
    add_message_to_history(session_id, 'user', user_input)

    # Display user message
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
                        'language': glanguage,
                        'messages': messages
                    }
                )

                # Add assistant's response to the session history
                add_message_to_history(session_id, 'assistant', response.content)

                # Display the response
                st.write(response.content)
            except Exception as e:
                st.error(f"An error occurred: {e}")
