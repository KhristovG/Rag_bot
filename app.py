import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.gigachat import GigaChat
from langchain.chains import create_retrieval_chain

from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain_text_splitters import CharacterTextSplitter


st.title("Q&A GigaChat")
#авторизация
with st.sidebar:
    st.title("GIGACHAT API")
    model_giga = st.selectbox(
        "GIGACHAT_MODEL",
        (
            "GigaChat",
            "GigaChat-Pro",
            "GigaChat-Plus",
        ),
    )
    st.title("GIGACHAT API")
    base_url = st.selectbox(
        "GIGACHAT_BASE_URL",
        (
            "https://gigachat.devices.sberbank.ru/api/v1",
            "https://beta.saluteai.sberdevices.ru/v1",
        ),
    )
    st.title("Авторизационные данные")
    scope = st.selectbox(
        "GIGACHAT_SCOPE",
        (
            "GIGACHAT_API_PERS",
            "GIGACHAT_API_CORP",
        ),
    )
    credentials = st.text_input("GIGACHAT_CREDENTIALS", type="password")

#загрузка файла
uploaded_file = st.file_uploader("Загрузите текст", type="pdf")

#вопрос
question = st.text_input(
    "Спросите что-нибудь про этот текст",
    placeholder="",
    disabled=not uploaded_file and not credentials,
)

#ссылка ютуб
youtube_url = st.text_input("ссылка на видеоролик с YouTube")
# st.write(youtube_url)

youtube_question = st.text_input(
    "Спросите что-нибудь про этот видеоролик",
    placeholder="",
    disabled=not youtube_url and not credentials,
)

if youtube_url and not credentials:
    st.info("Заполните данные GigaChat для того, чтобы продолжить")


if uploaded_file and not credentials:
    st.info("Заполните данные GigaChat для того, чтобы продолжить")


if uploaded_file and question:
    # разбиение текста на чанки
    ext = os.path.splitext(uploaded_file.name)[-1].lower()
    # st.write(ext)

    temp_file = f"./temp{ext}"
    with open(temp_file, "wb") as file:
        file.write(uploaded_file.getvalue())
        file_name = uploaded_file.name

    # if ext == '.txt':
    #     documents = uploaded_file.read().decode()
    # else:

    loader = PyPDFLoader(temp_file)
    # loader = PyPDFLoader(temp_file, extract_images=True)#если PDF в виде скана
    documents = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    splitted_data = text_splitter.split_documents(documents)
    st.write('текст разбит')

    os.remove(temp_file)

    #выбор эмбеддинговой модели
    #https://github.com/avidale/encodechka#%D0%BB%D0%B8%D0%B4%D0%B5%D1%80%D0%B1%D0%BE%D1%80%D0%B4
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding = HuggingFaceEmbeddings(model_name=model_name,
                                      model_kwargs=model_kwargs,
                                      encode_kwargs=encode_kwargs)
    #векторная бд
    vector_store = FAISS.from_documents(splitted_data, embedding=embedding)
    embedding_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    #создание модели gigachat
    llm = GigaChat(credentials=credentials,
                   model=model_giga,
                   verify_ssl_certs=False
                   )
    # prompt = ChatPromptTemplate.from_template('''Ответь на вопрос пользователя. \
    # Используй при этом только информацию из контекста. Если в контексте нет \
    # информации для ответа, сообщи об этом пользователю.
    # Контекст: {context}
    # Вопрос: {input}
    # Ответ:''')
    prompt = ChatPromptTemplate.from_template('''Ответь на вопрос пользователя. \
    Используй при этом только информацию из контекста. 
    Контекст: {context}
    Вопрос: {input}
    Ответ:''')

    #Создадим цепочку create_stuff_documents_chain, которая будет частью нашей вопросно-ответной цепочки.
    #Это нужно, чтобы подавать фрагменты текстов из векторной БД в промпт языковой модели.
    # Промпт представляет из себя форматированную строку, а франменты являются экземплярами класса Document.
    # Чтобы не писать код по извлечению атрибута page_content из Document,
    # используем цепочку create_stuff_documents_chain, где это автоматизировано.

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
        )
    retrieval_chain = create_retrieval_chain(embedding_retriever, document_chain)

    st.write("ОТВЕТ МОДЕЛИ")
    #     #ответ модели
    #vector_store.similarity_search(f'{question}', k=3)# похожие чанки
    resp = retrieval_chain.invoke({'input': question})
    st.write(resp['answer'])
    with st.expander("*Ответ на ваш вопрос может быть в одном из этих отрывков*"):
        st.write(vector_store.similarity_search(f'{question}?', k=3))

# YouTube

def load_youtube(url):
    loader_y = YoutubeLoader.from_youtube_url(
        youtube_url=url, #ссылка на видео
        add_video_info=False, #добавить описание
        transcript_format=TranscriptFormat.CHUNKS,
        chunk_size_seconds=30, #создание чанков по 30 сек.
        language=['ru'],
    )
    text_splitter_y = CharacterTextSplitter(
        separator="\n\n", #на основе чего мы будем делить
        chunk_size=1000, #сколько символов в одном кусочке
        chunk_overlap=100, #на сколько они будут накладывать друг на друга
        length_function=len,
        is_separator_regex=False,
    )
    return loader_y.load_and_split(text_splitter_y)


if youtube_url and credentials and youtube_question:
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding = HuggingFaceEmbeddings(model_name=model_name,
                                      model_kwargs=model_kwargs,
                                      encode_kwargs=encode_kwargs)

    vector_store = FAISS.from_documents(load_youtube(youtube_url), embedding=embedding)
    embedding_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = GigaChat(credentials=credentials,
                   model=model_giga,
                   verify_ssl_certs=False
                   )
    prompt = ChatPromptTemplate.from_template('''Ответь на вопрос пользователя. \
    Используй при этом только информацию из контекста. Если в контексте нет \
    информации для ответа, сообщи об этом пользователю.
    Контекст: {context}
    Вопрос: {input}
    Ответ:'''
    )
    #, по предоставленному сценарию видеоролика.

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )
    retrieval_chain = create_retrieval_chain(embedding_retriever, document_chain)

    st.write("<p style='text-align: center;'>ОТВЕТ МОДЕЛИ</p>", unsafe_allow_html=True)
    resp = retrieval_chain.invoke({'input': youtube_question})
    st.write(resp['answer'])
    with st.expander("*Ответ на ваш вопрос может быть в одном из этих отрывков*"):
        similar_quest = vector_store.similarity_search(f'{youtube_question}?', k=3)
        for i in [0, 1, 2]:
            st.markdown(f"<p style='text-align: center;'>Отрывок контекста №{i+1}</p>", unsafe_allow_html=True)
            st.write(similar_quest[i].page_content)
            st.markdown(f"<p style='text-align: center;'>Таймкод видеоролика №{i+1}</p>", unsafe_allow_html=True)
            st.write(similar_quest[i].metadata["source"])

