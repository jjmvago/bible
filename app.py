# Imports necessários
import os
import uuid
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, trim_messages
import streamlit as st

# Carregar variáveis de ambiente
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']=os.getenv('LANGCHAIN_TRACING_V2')
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT']=os.getenv('LANGCHAIN_PROJECT')

# Configuração do modelo de chat
llm = ChatGroq(model='Gemma2-9b-It', temperature=0.6)

# Configuração da página com ícone e título
st.set_page_config(page_title="The Bible Explorer", page_icon="📖")

# Exibir o título, cabeçalho e subtítulo no idioma selecionado
st.title("📖 The Bible Explorer")
st.markdown(f"<p style='font-size:16px;'>Explore biblical insights with AI.</p>", unsafe_allow_html=True)

import streamlit as st

# Configuração do Disclaimer com um expander
with st.expander("Disclaimer"):
    st.write("**The Bible Explorer – Uncover Christ in Every Verse** is a resource designed to support users in exploring Biblical texts with a focus on Christ-centered themes and insights. This application leverages artificial intelligence powered by open-source models, including *Gemma2-9b-It*, to generate thoughtful responses to user questions.")

    st.subheader("Please be aware of the following:")

    st.markdown("**1. Purpose and Interpretation**")
    st.write("The information provided by this app is generated through AI and is intended solely for educational and informational purposes. While the AI is trained to highlight biblical themes and historical context, it does not represent the official views or doctrine of any religious group.")

    st.markdown("**2. Accuracy of Content**")
    st.write("This app strives to provide helpful and accurate responses. However, the AI's answers are based on general Biblical knowledge and are not definitive interpretations. Users are encouraged to consult their own study and trusted spiritual advisors for guidance.")

    st.markdown("**3. Use of Open-Source Technology**")
    st.write("This app is powered by free, open-source AI technology, including models like *Gemma2-9b-It*. These tools are provided as-is, and we acknowledge that their content generation may not be fully exhaustive or error-free.")

    st.markdown("**4. Spiritual Guidance**")
    st.write("While this app is designed to assist in exploring and understanding the Bible, it should not be viewed as a replacement for personal study, religious practice, or discussions with knowledgeable spiritual mentors.")

    st.markdown("**5. User Responsibility**")
    st.write("By using this app, users acknowledge that AI-generated responses are provided as an additional resource and are not legally binding or definitive. Users are responsible for how they interpret and apply the insights provided.")

    st.markdown("**6. Privacy**")
    st.write("User privacy is a priority. Information shared within this app is used solely for generating relevant responses and enhancing the service experience. Please refer to our privacy policy for more details.")


# Gerar um ID de sessão único para cada usuário
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = str(uuid.uuid4())

user_id = st.session_state['user_id']

# Inicialização do estado de sessão específico para cada usuário
if f'messages_{user_id}' not in st.session_state:
    st.session_state[f'messages_{user_id}'] = [
        {'role': 'system', 'content': 'You are a chatbot specializing in biblical studies who helps people to see Jesus in all bible. With in-depth knowledge of Bible history and Christian life give for user a good powerful explanation, connecting the topic with Jesus in new Testament. Also, share some references at end.Try to share some verses to help the user. Identify the laguange before answering.'},
        {'role': 'assistant', 'content': "Hi, ask me a question about Bible' context and Jesus Christ!"}
    ]
st.session_state[f'chat_engine_{user_id}'] = llm

# Entrada do usuário
if prompt := st.chat_input('Your question'):
    st.session_state[f'messages_{user_id}'].append({'role': 'user', 'content': prompt})

# Exibir o histórico de mensagens, exceto as do sistema
for message in st.session_state[f'messages_{user_id}']:
    if message['role'] != 'system':  # Exclui mensagens de sistema
        with st.chat_message(message['role']):
            st.write(message['content'])

# Processamento da resposta do assistente usando `invoke` com trim
if st.session_state[f'messages_{user_id}'][-1]['role'] == 'user':
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            # Converter mensagens de usuário para `HumanMessage` e aplicar trim
            all_messages = [
                HumanMessage(content=msg['content']) if msg['role'] == 'user' 
                else HumanMessage(content=msg['content'], role="assistant")
                for msg in st.session_state[f'messages_{user_id}'] if msg['role'] != 'system'
            ]

            # Aplicar trim_messages para manter o histórico dentro do limite
            trimmed_messages = trim_messages(all_messages, token_counter=llm, max_tokens=1000)  # Ajuste max_tokens conforme necessário
            
            # Geração da resposta usando o método `invoke`
            response = st.session_state[f'chat_engine_{user_id}'].invoke(trimmed_messages)
            response_content = response.content  # Armazenar o conteúdo da resposta
            
            # Exibir e armazenar a resposta
            st.write(response_content)
            st.session_state[f'messages_{user_id}'].append({'role': 'assistant', 'content': response_content})
