# Utilidades del sistema
import os

# Bibliotecas externas
import streamlit as st

# Pinecone y módulos relacionados
from pinecone import Pinecone, ServerlessSpec

# LangChain y módulos relacionados
import langchain_community
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone as PineconeVectorStore


# ---------------------------------------------------------
# Obtiene las claves API necesarias desde las variables de entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = "pr-plaintive-radar-87"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

assert OPENAI_API_KEY is not None, "Falta la variable de entorno OPENAI_API_KEY"
assert LANGCHAIN_API_KEY is not None, "Falta la variable de entorno LANGCHAIN_API_KEY"
assert LANGCHAIN_TRACING_V2 is not None, "Falta la variable de entorno LANGCHAIN_TRACING_V2"
assert PINECONE_API_KEY is not None, "Falta la variable de entorno PINECONE_API_KEY"

# ---------------------------------------------------------

# Inicializa el modelo de chat de OpenAI con configuración personalizada
chat = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model='gpt-3.5-turbo',
    temperature=0,
    streaming=True
)

# EMBEDDINGS
# Ajustamos el modelo de embeddings
embed_model = OpenAIEmbeddings(model="text-embedding-3-large")

# ---------------------------------------------------------

namespace = "Curriculums"


# Crear instancia Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = 'llm-tp1'

# Obtener el índice
index = pc.Index(index_name)

# Crear vector stores
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embed_model,
    text_key="text",
    namespace=namespace
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# ---------------------------------------------------------
# Configurar la página de Streamlit
st.set_page_config(page_title="Chat", page_icon="💬")

st.title("Chat")

# Inicializar el estado de la sesión para almacenar el historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Campo de entrada para el usuario
user_query = st.text_input("Tu pregunta:", placeholder="Escribe tu pregunta aquí y presiona Enter...")

if user_query:  # Procesar solo si hay entrada del usuario
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=retriever
    )
    # Ejecutar la consulta
    result = qa_chain.run(user_query)

    # Agregar la respuesta al historial
    st.session_state.messages.append(("Usuario", user_query))
    st.session_state.messages.append(("Asistente", result))

# Mostrar el historial en la interfaz
for role, message in st.session_state.messages:
    if role == "Usuario":
        st.markdown(f"**{role}:** {message}")
    else:
        st.markdown(f"**{role}:** {message}")
