##################################################################
# GPT3 based Q&A model
# 
# History
# When      | Who            | What
# 15/04/2023| TQ Ye          | Created (v0.01)
##################################################################
import os
import sys
import shutil
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders import PyPDFLoader
import streamlit as st
from typing import List


#!import nltk
#!nltk.download('all')


# Define the folder path where your documents are
DATA_PATH = "./tempDir"

class Locale:    
    ai_role_options: List[str]
    ai_role_prefix: str
    ai_role_postfix: str
    title: str
    language: str
    lang_code: str
    chat_placeholder: str
    chat_run_btn: str
    chat_clear_btn: str
    chat_save_btn: str
    select_placeholder1: str
    select_placeholder2: str
    select_placeholder3: str
    radio_placeholder: str
    radio_text1: str
    radio_text2: str
    stt_placeholder: str
    
    def __init__(self, 
                ai_role_options, 
                ai_role_prefix,
                ai_role_postfix,
                title,
                language,
                lang_code,
                chat_placeholder,
                chat_run_btn,
                chat_clear_btn,
                chat_save_btn,
                select_placeholder1,
                select_placeholder2,
                select_placeholder3,
                radio_placeholder,
                radio_text1,
                radio_text2,
                stt_placeholder,
                
                ):
        self.ai_role_options = ai_role_options, 
        self.ai_role_prefix= ai_role_prefix,
        self.ai_role_postfix= ai_role_postfix,
        self.title= title,
        self.language= language,
        self.lang_code= lang_code,
        self.chat_placeholder= chat_placeholder,
        self.chat_run_btn= chat_run_btn,
        self.chat_clear_btn= chat_clear_btn,
        self.chat_save_btn= chat_save_btn,
        self.select_placeholder1= select_placeholder1,
        self.select_placeholder2= select_placeholder2,
        self.select_placeholder3= select_placeholder3,
        self.radio_placeholder= radio_placeholder,
        self.radio_text1= radio_text1,
        self.radio_text2= radio_text2,
        self.stt_placeholder= stt_placeholder,


AI_ROLE_OPTIONS_EN = [
    "helpful assistant",
    "code assistant",
    "code reviewer",
    "text improver",
    "english teacher",
    "sports expert",
]

AI_ROLE_OPTIONS_ZW = [
    "helpful assistant",
    "code assistant",
    "code reviewer",
    "text improver",
    "english teacher",
    "sports expert",
]

en = Locale(
    ai_role_options=AI_ROLE_OPTIONS_EN,
    ai_role_prefix="You are an assistant",
    ai_role_postfix="Answer as concisely as possible.",
    title="Ask Your Docs",
    language="English",
    lang_code="en",
    chat_placeholder="Your Question:",
    chat_run_btn="Ask",
    chat_clear_btn="Clear",
    chat_save_btn="Save",
    select_placeholder1="Select Model",
    select_placeholder2="Select Role",
    select_placeholder3="Create Role",
    radio_placeholder="Role Interaction",
    radio_text1="Select",
    radio_text2="Create",
    stt_placeholder="To Hear The Voice Of AI Press Play",
)
    
def get_docx_data(filepath):
    '''
    File types: docx
    '''
    loader = UnstructuredWordDocumentLoader(filepath)

    data = loader.load()
    doc = data[0]

    return doc

def get_ppt_data(filepath):
    '''
    File types: powerpoint document
    '''
    loader = UnstructuredPowerPointLoader(filepath)
    docs = loader.load()
    doc = docs[0]

    file_name = os.path.basename(filepath)
    doc.metadata["source"] = file_name

    '''
    [Document(page_content='Adding a Bullet Slide\n\nFind the bullet slide layout\n\nUse _TextFrame.text for first bullet\n\nUse _TextFrame.add_paragraph() for subsequent bullets\n\nHere is a lot of text!\n\nHere is some text in a text box!', lookup_str='', metadata={'source': 'example_data/fake-power-point.pptx'}, lookup_index=0)]
    '''
    return doc

def get_pdf_data(filepath):
    '''
    File types: pdf
    '''
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    doc = docs[0]

    file_name = os.path.basename(filepath)
    doc.metadata["source"] = file_name
    return doc

def get_unstructured_data(filepath):
    '''
    File types: text, html, pdf
    '''
    loader = UnstructuredFileLoader(filepath)
    docs = loader.load()
    doc = docs[0]

    file_name = os.path.basename(filepath)
    doc.metadata["source"] = file_name
    return doc

def text_preprocessing(filepath):
    '''
    Readin and Preprocessing training data
    '''
    with open(filepath, encoding="utf-8") as f:
        magbasics = f.read()

    #--- Leave this splitting to later -----
    # text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    # texts = text_splitter.split_text(magbasics)
    #doc = Document(page_content=texts, metadata={"source": f"{filepath}"},)
    file_name = os.path.basename(filepath)
    doc = Document(page_content=magbasics, metadata={"source": f"{file_name}"},)

    return doc
     
def save_log(text) -> None:
    now = datetime.now() # current date and time
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    
    f = open("QA.log", "a", encoding='utf-8',)
    f.write(f'[{date_time}]: {text}\n\n')
    f.close()

def Clear_Chat() -> None:
    st.session_state.generated = []
    st.session_state.past = []
    st.session_state.messages = []
    st.session_state.user_text = ""

def show_text_input() -> None:
    st.text_area(label=st.session_state.locale.chat_placeholder, value=st.session_state.user_text, key="user_text")

def create_gpt_completion(ai_model: str, messages: List[dict]) -> dict:
    openai.api_key = st.secrets.api_credentials.api_key
    completion = openai.ChatCompletion.create(
        model=ai_model,
        messages=messages,
    )
    return completion
    
@st.cache_data
def show_chat(ai_content: str, user_text: str) -> None:
    if ai_content not in st.session_state.generated:
        # store the ai content
        st.session_state.past.append(user_text)
        st.session_state.generated.append(ai_content)
    if st.session_state.generated:
        for i in range(len(st.session_state.generated)):
            message(st.session_state.past[i], is_user=True, key=str(i) + "_user", avatar_style="micah")
            message("", key=str(i))
            st.markdown(st.session_state.generated[i])
            
def show_gpt_conversation() -> None:
    try:
        completion = create_gpt_completion(st.session_state.model, st.session_state.messages)
        ai_content = completion.get("choices")[0].get("message").get("content")
        st.session_state.messages.append({"role": "assistant", "content": ai_content})
        if ai_content:
            show_chat(ai_content, st.session_state.user_text)
            st.divider()
            show_audio_player(ai_content)
    except InvalidRequestError as err:
        if err.code == "context_length_exceeded":
            st.session_state.messages.pop(1)
            if len(st.session_state.messages) == 1:
                st.session_state.user_text = ""
            show_conversation()
        else:
            st.error(err)
    except (OpenAIError, UnboundLocalError) as err:
        st.error(err)
        
def show_conversation() -> None:
    if st.session_state.messages:
        st.session_state.messages.append({"role": "user", "content": st.session_state.user_text})
    else:
        ai_role = f"{st.session_state.locale.ai_role_prefix} {st.session_state.role}. {st.session_state.locale.ai_role_postfix}"  # NOQA: E501
        st.session_state.messages = [
            {"role": "system", "content": ai_role},
            {"role": "user", "content": st.session_state.user_text},
        ]
    show_gpt_conversation()

@st.cache_data
def save_uploadedfile(uploadedfile) -> None:

    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    with open(os.path.join(DATA_PATH, uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())

     #return st.success("Saved File:{} to tempDir".format(uploadedfile.name))

@st.cache_data(experimental_allow_widgets=True) 
def Get_API_Key() -> str:
    openai_key = st.sidebar.text_input('Your OpenAI Key:')
    return openai_key


def Load_Files():
    uploaded_files = st.file_uploader("Load your file(s)", type=['docx', 'txt', 'pdf'], accept_multiple_files=True)
    count = 0
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            bytes_data = uploaded_file.read()
            file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
            save_uploadedfile(uploaded_file)
            count += 1

        # st.write("filename:", uploaded_file.name)
        # st.write(bytes_data)
    
    nfiles = len(os.listdir(DATA_PATH))
    return nfiles
    

def Delete_Files(folder = DATA_PATH) -> None:
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    st.session_state['new_file_loaded'] = False


@st.cache_resource
def Build_Search_Index(docPath, files):    
    sources = []
    for file in files:
        file = os.path.join(docPath, file)
        print(file)
        if file.split(".")[-1] == 'txt':
            sources.append(get_unstructured_data(file))
        elif file.split(".")[-1] == 'docx':
            sources.append(get_docx_data(file))
        elif file.split(".")[-1] == 'pdf':
            sources.append(get_pdf_data(file))
            
    #sources = [
        # #get_unstructured_data(datapath + '/' + "tqWiKi.txt"),
        # #text_preprocessing(datapath + '/' + "tqWiKi.txt"),
        # #get_unstructured_data(datapath + '/' + "SLiM_API.html"),
        # get_pdf_data(DATA_PATH + '/' + "ParallelUniverses.pdf"),
        # get_pdf_data(DATA_PATH + '/' + "FreezeLayersTransformerFT.pdf"),
        # # get_wiki_data("Microsoft_Windows", True),
    #]
    
    source_chunks = []
    splitter = CharacterTextSplitter(separator="\n\n", chunk_size=512, chunk_overlap=100)
    for source in sources:
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

    # do some embeddings search
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings()
    search_index = FAISS.from_documents(source_chunks, embeddings)

    st.session_state['new_file_loaded'] = False  #All docs are indexed.

    return search_index

@st.cache_resource
def Create_Model_Chain():

    template = """Given the following document and a question, create an answer with references ("SOURCES"). 
                If you don't know the answer, just say you don't know.

                QUESTION: {question}
                {summaries}
                ANSWER:"""

    PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])

    chat_model = ChatOpenAI(temperature=0, max_tokens=256)  # --> gpt-3.5-turbo
    chain = load_qa_with_sources_chain(chat_model, chain_type="stuff", prompt=PROMPT)
    #chain = load_qa_with_sources_chain(OpenAI(model_name="text-davinci-003", temperature=0, max_tokens=256), chain_type="stuff", prompt=PROMPT)
    #chain = load_qa_with_sources_chain(ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=256), chain_type="stuff", prompt=PROMPT)

    return chain


##############################################
################ MAIN ########################
##############################################
def main(argv):

    # Get user supplied key
    os.environ["OPENAI_API_KEY"] = Get_API_Key()

    nfiles = Load_Files()
    if nfiles > 0:
        st.session_state['new_file_loaded'] = True
        st.success("File loading done")
        
    if st.session_state['new_file_loaded'] == False:
        return 

    # Remove documents from the temp folder
    c1, c2 = st.columns(2)
    with c1:
        st.button("Clear Documents", on_click=Delete_Files)
    with c2:
        st.button("Clear Messages", on_click=Clear_Chat)

    # Build search Index
    files = os.listdir(DATA_PATH)
    search_index = Build_Search_Index(DATA_PATH, files)
    
    # Check if API Key is available
    if os.environ["OPENAI_API_KEY"] == "":
        st.write("Please provide your OpenAI API Key!")
        return

    ## Build Model Chain
    try:
        with st.spinner('Wait ...'):
            chain = Create_Model_Chain()
            #print("Build model chain done!")
    except Exception as e:
        st.write(e)
        return

    inpit_placeholder = st.empty()
    history = []
    user_input = inpit_placeholder.text_input(label="Your query:", value=st.session_state.user_text, max_chars=256, key="1")
    send_button = st.button("Send")
    if send_button:
        if(user_input.strip() != ''):
            query = user_input.strip()
            history.append(f"You: {query}")            
            docs = search_index.similarity_search_with_score(query, k=4) 
            good_docs = []
            index = 0
            save_log(f"Query: {query}\n Contexts:")
            for doc, score in docs:
    #            good_docs.append(doc)
                if index == 0:
                    good_docs.append(doc) # always include the first one into it
                    save_log(f"Included: [{score}] {doc}")
                    index = index + 1
                else:
                    if score < 0.9:        # only include those whose score value is less than 1.0
                        print(f"Included: [{score}] {doc}")
                        good_docs.append(doc)
                        save_log(f"Included: [{score}] {doc}")
                    else:
                        print(f"Ignored: [{score}] {doc}")
                        save_log(f"Ignored: [{score}] {doc}")

            ## Langchain must have bug here. We need to pass at least 2 docs to the chain,
            ## so that it can trigger search externally, i.e. search from out of context sources for answer.
            #if len(good_docs) < 2:
            #    print("Add a dummy doc")
            #    dummy_doc = Document(page_content="---", metadata={"source": ""},)
            #    good_docs.append(dummy_doc)

            print (f"good_docs: {len(good_docs)}")

            with st.spinner('Wait ...'):
                results = chain({"input_documents": good_docs, "question": query}, return_only_outputs=True)
            answer = results["output_text"]
            if 0 < answer.find("I don't know") or 0 < answer.find("I do not know"):
                #No answer found. Try another way once
                #print("-->No answer found from the context! Try another way ...")
                docs = search_index.similarity_search(query, k=3)
                save_log("\n\n--No answer found from the context! Try another way--\n")
                save_log("Include All")

                with st.spinner('Wait ...'):
                    results = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
                answer = results["output_text"]

            # displaying messages
            st.session_state.messages.append({"you": query, "content": answer})
            for msg in reversed(st.session_state.messages):
                st.write(f"You: {msg['you']}")
                st.success(f"Bot: {msg['content']}")
                
            history.append(f"Bot: {answer}")
            save_log("\n".join(history))

        # clear the input field
        user_input = inpit_placeholder.text_input(label="Your query:", value="", max_chars=256, key="2")


##############################
# Program Start
##############################
if __name__ == "__main__":

    # Initialising session state
    if 'new_file_loaded' not in st.session_state:
        st.session_state['new_file_loaded'] = False

    if "locale" not in st.session_state:
        st.session_state.locale = en

    if "generated" not in st.session_state:
        st.session_state.generated = []

    if "past" not in st.session_state:
        st.session_state.past = []

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "user_text" not in st.session_state:
        st.session_state.user_text = ""

    st.markdown(f"<h1 style='text-align: center;'>{st.session_state.locale.title[0]}</h1>", unsafe_allow_html=True)

    main(sys.argv)
    
