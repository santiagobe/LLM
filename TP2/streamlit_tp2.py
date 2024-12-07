# Utilidades del sistema
import os
import time
import re

# Bibliotecas externas
import streamlit as st
from tqdm.auto import tqdm

# Pinecone y módulos relacionados
from pinecone import Pinecone, ServerlessSpec

# Procesamiento de documentos
from odf.opendocument import load
from odf.text import P

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
# Ajustamos el modelo de embeddings a uno conocido: text-embedding-ada-002 (dim=1536)
embed_model = OpenAIEmbeddings(model="text-embedding-3-large")

# ---------------------------------------------------------

namespace_Belen = "espacio_Belen"
namespace_Deshays = "espacio_Deshays"

# Crear instancia Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = 'llm-tp2'

# Obtener el índice
index = pc.Index(index_name)

# Crear vector stores
vectorstore_Belen = PineconeVectorStore(
    index=index,
    embedding=embed_model,
    text_key="text",
    namespace=namespace_Belen
)
retriever_Belen = vectorstore_Belen.as_retriever()


vectorstore_Deshays = PineconeVectorStore(
    index=index,
    embedding=embed_model,
    text_key="text",
    namespace=namespace_Deshays
)
retriever_Deshays = vectorstore_Deshays.as_retriever()

# ---------------------------------------------------------
# Función para decidir el namespace según la pregunta
def decidir_namespace(pregunta: str):
    pregunta_lower = pregunta.lower()
    # Chequear palabras clave
    if re.search(r"\bsantiago\b", pregunta_lower) or re.search(r"\bbelen\b", pregunta_lower):
        return "espacio_Belen"
    elif re.search(r"\boctavio\b", pregunta_lower) or re.search(r"\bdeshays\b", pregunta_lower):
        return "espacio_Deshays"
    else:
        # Por defecto, si no se mencionan las palabras, usar espacio_Belen
        return "espacio_Belen"

# ---------------------------------------------------------
# Configurar la página de Streamlit
st.set_page_config(page_title="Chat con Decisor Regex", page_icon="💬")

st.title("Chat con Decisor Regex")

# Inicializar el estado de la sesión para almacenar el historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Campo de entrada para el usuario
user_query = st.text_input("Tu pregunta:", placeholder="Escribe tu pregunta aquí y presiona Enter...")

# Cuando el usuario introduce una pregunta
if user_query:
    # Agregar la pregunta del usuario al historial
    st.session_state.messages.append(("Usuario", user_query))

    # Decidir el namespace a consultar
    namespace = decidir_namespace(user_query)

    # Seleccionar el retriever según el namespace
    if namespace == "espacio_Belen":
        current_retriever = retriever_Belen
    else:
        current_retriever = retriever_Deshays

    # Crear la cadena QA con el retriever elegido
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=current_retriever
    )

    # Ejecutar la consulta
    result = qa_chain.run(user_query)

    # Agregar la respuesta al historial
    st.session_state.messages.append(("Asistente", result))

# Mostrar el historial en la interfaz
for role, message in st.session_state.messages:
    if role == "Usuario":
        st.markdown(f"**{role}:** {message}")
    else:
        st.markdown(f"**{role}:** {message}")
