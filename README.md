# AI Pipelines with LLM
Large Language Model Demonstrations built to teach latest SOTA in AI Pipelines as of 9/7/2023

# AI Pair Programming   -   Design for Transparency - Tooltips and Markdown
```
# Imports
import base64
import glob
import json
import math
import mistune
import openai
import os
import pytz
import re
import requests
import streamlit as st
import textract
import time
import zipfile
from audio_recorder_streamlit import audio_recorder
from bs4 import BeautifulSoup
from collections import deque
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from io import BytesIO
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from openai import ChatCompletion
from PyPDF2 import PdfReader
from templates import bot_template, css, user_template
from xml.etree import ElementTree as ET

# Constants
API_URL = 'https://qe55p8afio98s0u3.us-east-1.aws.endpoints.huggingface.cloud'  # Dr Llama
API_KEY = os.getenv('API_KEY')
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
key = os.getenv('OPENAI_API_KEY')
prompt = f"Write instructions to teach anyone to write a discharge plan. List the entities, features and relationships to CCDA and FHIR objects in boldface."
# page config and sidebar declares up front allow all other functions to see global class variables
st.set_page_config(page_title="GPT Streamlit Document Reasoner", layout="wide")

# UI Controls
should_save = st.sidebar.checkbox("💾 Save", value=True, help="Save your session data.")

# Function to add witty and humor buttons
def add_witty_humor_buttons():
    with st.expander("Wit and Humor 🤣", expanded=True, help="Expand to access various humor features."):
        # Tip about the Dromedary family
        st.markdown("🔬 **Fun Fact**: Dromedaries, part of the camel family, have a single hump and are adapted to arid environments. Their 'superpowers' include the ability to survive without water for up to 7 days, thanks to their specialized blood cells and water storage in their hump.")
        
        # Define button descriptions
        descriptions = {
            "Generate Limericks 😂": "Write ten random adult limericks based on quotes that are tweet length and make you laugh 🎭",
            "Wise Quotes 🧙": "Generate ten wise quotes that are tweet length 🦉",
            "Funny Rhymes 🎤": "Create ten funny rhymes that are tweet length 🎶",
            "Medical Jokes 💉": "Create ten medical jokes that are tweet length 🏥",
            "Minnesota Humor ❄️": "Create ten jokes about Minnesota that are tweet length 🌨️",
            "Top Funny Stories 📖": "Create ten funny stories that are tweet length 📚",
            "More Funny Rhymes 🎙️": "Create ten more funny rhymes that are tweet length 🎵"
        }
        
        # Create columns
        col1, col2, col3 = st.columns([1, 1, 1], gap="small")
        
        # Add buttons to columns
        if col1.button("Generate Limericks 😂"):
            StreamLLMChatResponse(descriptions["Generate Limericks 😂"])
        
        if col2.button("Wise Quotes 🧙"):
            StreamLLMChatResponse(descriptions["Wise Quotes 🧙"])
        
        if col3.button("Funny Rhymes 🎤"):
            StreamLLMChatResponse(descriptions["Funny Rhymes 🎤"])
        
        col4, col5, col6 = st.columns([1, 1, 1], gap="small")
        
        if col4.button("Medical Jokes 💉"):
            StreamLLMChatResponse(descriptions["Medical Jokes 💉"])
        
        if col5.button("Minnesota Humor ❄️"):
            StreamLLMChatResponse(descriptions["Minnesota Humor ❄️"])
        
        if col6.button("Top Funny Stories 📖"):
            StreamLLMChatResponse(descriptions["Top Funny Stories 📖"])
        
        col7 = st.columns(1, gap="small")
        
        if col7[0].button("More Funny Rhymes 🎙️"):
            StreamLLMChatResponse(descriptions["More Funny Rhymes 🎙️"])


# Function to Stream Inference Client for Inference Endpoint Responses
def StreamLLMChatResponse(prompt):
    endpoint_url = API_URL
    hf_token = API_KEY
    client = InferenceClient(endpoint_url, token=hf_token)
    gen_kwargs = dict(
        max_new_tokens=512,
        top_k=30,
        top_p=0.9,
        temperature=0.2,
        repetition_penalty=1.02,
        stop_sequences=["\nUser:", "<|endoftext|>", "</s>"],
    )
    stream = client.text_generation(prompt, stream=True, details=True, **gen_kwargs)
    report=[]
    res_box = st.empty()
    collected_chunks=[]
    collected_messages=[]
    for r in stream:
        if r.token.special:
            continue
        if r.token.text in gen_kwargs["stop_sequences"]:
            break
        collected_chunks.append(r.token.text)
        chunk_message = r.token.text
        collected_messages.append(chunk_message)
        try:
            report.append(r.token.text)
            if len(r.token.text) > 0:
                result="".join(report).strip()
                res_box.markdown(f'*{result}*')
        except:
            st.write(' ')

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    st.markdown(response.json())
    return response.json()

def get_output(prompt):
    return query({"inputs": prompt})

def generate_filename(prompt, file_type):
    central = pytz.timezone('US/Central')
    safe_date_time = datetime.now(central).strftime("%m%d_%H%M")
    replaced_prompt = prompt.replace(" ", "_").replace("\n", "_")
    safe_prompt = "".join(x for x in replaced_prompt if x.isalnum() or x == "_")[:90]
    return f"{safe_date_time}_{safe_prompt}.{file_type}"

def transcribe_audio(openai_key, file_path, model):
    openai.api_key = openai_key
    OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {openai_key}",
    }
    with open(file_path, 'rb') as f:
        data = {'file': f}
        response = requests.post(OPENAI_API_URL, headers=headers, files=data, data={'model': model})
    if response.status_code == 200:
        st.write(response.json())
        chatResponse = chat_with_model(response.json().get('text'), '') # *************************************
        transcript = response.json().get('text')
        filename = generate_filename(transcript, 'txt')
        response = chatResponse
        user_prompt = transcript
        create_file(filename, user_prompt, response, should_save)
        return transcript
    else:
        st.write(response.json())
        st.error("Error in API call.")
        return None

def save_and_play_audio(audio_recorder):
    audio_bytes = audio_recorder()
    if audio_bytes:
        filename = generate_filename("Recording", "wav")
        with open(filename, 'wb') as f:
            f.write(audio_bytes)
        st.audio(audio_bytes, format="audio/wav")
        return filename
    return None

def create_file(filename, prompt, response, should_save=True):
    if not should_save:
        return
    base_filename, ext = os.path.splitext(filename)
    has_python_code = bool(re.search(r"```python([\s\S]*?)```", response))
    if ext in ['.txt', '.htm', '.md']:
        with open(f"{base_filename}-Prompt.txt", 'w') as file:
            file.write(prompt)
        with open(f"{base_filename}-Response.md", 'w') as file:
            file.write(response)
        if has_python_code:
            python_code = re.findall(r"```python([\s\S]*?)```", response)[0].strip()
            with open(f"{base_filename}-Code.py", 'w') as file:
                file.write(python_code)
            
def truncate_document(document, length):
    return document[:length]

def divide_document(document, max_length):
    return [document[i:i+max_length] for i in range(0, len(document), max_length)]

def get_table_download_link(file_path):
    with open(file_path, 'r') as file:
        try:
            data = file.read()
        except:
            st.write('')
            return file_path    
    b64 = base64.b64encode(data.encode()).decode()  
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1]  # get the file extension
    if ext == '.txt':
        mime_type = 'text/plain'
    elif ext == '.py':
        mime_type = 'text/plain'
    elif ext == '.xlsx':
        mime_type = 'text/plain'
    elif ext == '.csv':
        mime_type = 'text/plain'
    elif ext == '.htm':
        mime_type = 'text/html'
    elif ext == '.md':
        mime_type = 'text/markdown'
    else:
        mime_type = 'application/octet-stream'  # general binary data type
    href = f'<a href="data:{mime_type};base64,{b64}" target="_blank" download="{file_name}">{file_name}</a>'
    return href

def CompressXML(xml_text):
    root = ET.fromstring(xml_text)
    for elem in list(root.iter()):
        if isinstance(elem.tag, str) and 'Comment' in elem.tag:
            elem.parent.remove(elem)
    return ET.tostring(root, encoding='unicode', method="xml")
    
def read_file_content(file,max_length):
    if file.type == "application/json":
        content = json.load(file)
        return str(content)
    elif file.type == "text/html" or file.type == "text/htm":
        content = BeautifulSoup(file, "html.parser")
        return content.text
    elif file.type == "application/xml" or file.type == "text/xml":
        tree = ET.parse(file)
        root = tree.getroot()
        xml = CompressXML(ET.tostring(root, encoding='unicode'))
        return xml
    elif file.type == "text/markdown" or file.type == "text/md":
        md = mistune.create_markdown()
        content = md(file.read().decode())
        return content
    elif file.type == "text/plain":
        return file.getvalue().decode()
    else:
        return ""

def chat_with_model(prompt, document_section, model_choice='gpt-3.5-turbo'):
    model = model_choice
    conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    conversation.append({'role': 'user', 'content': prompt})
    if len(document_section)>0:
        conversation.append({'role': 'assistant', 'content': document_section})
    start_time = time.time()
    report = []
    res_box = st.empty()
    collected_chunks = []
    collected_messages = []
    for chunk in openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=conversation, temperature=0.5, stream=True):
        collected_chunks.append(chunk)  
        chunk_message = chunk['choices'][0]['delta']  
        collected_messages.append(chunk_message) 
        content=chunk["choices"][0].get("delta",{}).get("content")
        try:
            report.append(content)
            if len(content) > 0:
                result = "".join(report).strip()
                res_box.markdown(f'*{result}*') 
        except:
            st.write(' ')
    full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
    st.write("Elapsed time:")
    st.write(time.time() - start_time)
    return full_reply_content

def chat_with_file_contents(prompt, file_content, model_choice='gpt-3.5-turbo'):
    conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    conversation.append({'role': 'user', 'content': prompt})
    if len(file_content)>0:
        conversation.append({'role': 'assistant', 'content': file_content})
    response = openai.ChatCompletion.create(model=model_choice, messages=conversation)
    return response['choices'][0]['message']['content']

def extract_mime_type(file):
    if isinstance(file, str):
        pattern = r"type='(.*?)'"
        match = re.search(pattern, file)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Unable to extract MIME type from {file}")
    elif isinstance(file, streamlit.UploadedFile):
        return file.type
    else:
        raise TypeError("Input should be a string or a streamlit.UploadedFile object")

def extract_file_extension(file):
    # get the file name directly from the UploadedFile object
    file_name = file.name
    pattern = r".*?\.(.*?)$"
    match = re.search(pattern, file_name)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Unable to extract file extension from {file_name}")

def pdf2txt(docs):
    text = ""
    for file in docs:
        file_extension = extract_file_extension(file)
        st.write(f"File type extension: {file_extension}")
        try:
            if file_extension.lower() in ['py', 'txt', 'html', 'htm', 'xml', 'json']:
                text += file.getvalue().decode('utf-8')
            elif file_extension.lower() == 'pdf':
                from PyPDF2 import PdfReader
                pdf = PdfReader(BytesIO(file.getvalue()))
                for page in range(len(pdf.pages)):
                    text += pdf.pages[page].extract_text() # new PyPDF2 syntax
        except Exception as e:
            st.write(f"Error processing file {file.name}: {e}")
    return text

def txt2chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

def vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=key)
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

def process_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        filename = generate_filename(user_question, 'txt')
        response = message.content
        user_prompt = user_question
        create_file(filename, user_prompt, response, should_save)       

def divide_prompt(prompt, max_length):
    words = prompt.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        if len(word) + current_length <= max_length:
            current_length += len(word) + 1 
            current_chunk.append(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
    chunks.append(' '.join(current_chunk))
    return chunks

def create_zip_of_files(files):
    zip_name = "all_files.zip"
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for file in files:
            zipf.write(file)
    return zip_name

def get_zip_download_link(zip_file):
    with open(zip_file, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/zip;base64,{b64}" download="{zip_file}">Download All</a>'
    return href

def main():

    st.title("DromeLlama7B")
    prompt = f"Write ten funny jokes that are tweet length stories that make you laugh.  Show as markdown outline with emojis for each."

    # Add Wit and Humor buttons
    add_witty_humor_buttons()

    example_input = st.text_input("Enter your example text:", value=prompt, help="Enter text to get a response from DromeLlama.")
    if st.button("Run Prompt With DromeLlama", help="Click to run the prompt."):
        try:
            StreamLLMChatResponse(example_input)
        except:
            st.write('DromeLlama is asleep. Starting up now on A10 - please give 5 minutes then retry as KEDA scales up from zero to activate running container(s).')

    
    openai.api_key = os.getenv('OPENAI_KEY')
    menu = ["txt", "htm", "xlsx", "csv", "md", "py"]
    choice = st.sidebar.selectbox("Output File Type:", menu)
    model_choice = st.sidebar.radio("Select Model:", ('gpt-3.5-turbo', 'gpt-3.5-turbo-0301'))
    filename = save_and_play_audio(audio_recorder)
    if filename is not None:
        transcription = transcribe_audio(key, filename, "whisper-1")
        st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)
        filename = None
    user_prompt = st.text_area("Enter prompts, instructions & questions:", '', height=100)
    collength, colupload = st.columns([2,3])  # adjust the ratio as needed
    with collength:
        max_length = st.slider("File section length for large files", min_value=1000, max_value=128000, value=12000, step=1000)
    with colupload:
        uploaded_file = st.file_uploader("Add a file for context:", type=["pdf", "xml", "json", "xlsx", "csv", "html", "htm", "md", "txt"])
    document_sections = deque()
    document_responses = {}
    if uploaded_file is not None:
        file_content = read_file_content(uploaded_file, max_length)
        document_sections.extend(divide_document(file_content, max_length))
    if len(document_sections) > 0:
        if st.button("👁️ View Upload"):
            st.markdown("**Sections of the uploaded file:**")
            for i, section in enumerate(list(document_sections)):
                st.markdown(f"**Section {i+1}**\n{section}")
        st.markdown("**Chat with the model:**")
        for i, section in enumerate(list(document_sections)):
            if i in document_responses:
                st.markdown(f"**Section {i+1}**\n{document_responses[i]}")
            else:
                if st.button(f"Chat about Section {i+1}"):
                    st.write('Reasoning with your inputs...')
                    response = chat_with_model(user_prompt, section, model_choice)
                    st.write('Response:')
                    st.write(response)
                    document_responses[i] = response
                    filename = generate_filename(f"{user_prompt}_section_{i+1}", choice)
                    create_file(filename, user_prompt, response, should_save)
                    st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)
    if st.button('💬 Chat'):
        st.write('Reasoning with your inputs...')
        user_prompt_sections = divide_prompt(user_prompt, max_length)
        full_response = ''
        for prompt_section in user_prompt_sections:
            response = chat_with_model(prompt_section, ''.join(list(document_sections)), model_choice)
            full_response += response + '\n'  # Combine the responses
        response = full_response
        st.write('Response:')
        st.write(response)
        filename = generate_filename(user_prompt, choice)
        create_file(filename, user_prompt, response, should_save)
        st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)
    all_files = glob.glob("*.*")
    all_files = [file for file in all_files if len(os.path.splitext(file)[0]) >= 20]  # exclude files with short names
    all_files.sort(key=lambda x: (os.path.splitext(x)[1], x), reverse=True)  # sort by file type and file name in descending order
    if st.sidebar.button("🗑 Delete All"):
        for file in all_files:
            os.remove(file)
        st.experimental_rerun()
    if st.sidebar.button("⬇️ Download All"):
        zip_file = create_zip_of_files(all_files)
        st.sidebar.markdown(get_zip_download_link(zip_file), unsafe_allow_html=True)
    file_contents=''
    next_action=''
    for file in all_files:
        col1, col2, col3, col4, col5 = st.sidebar.columns([1,6,1,1,1])  # adjust the ratio as needed
        with col1:
            if st.button("🌐", key="md_"+file):  # md emoji button
                with open(file, 'r') as f:
                    file_contents = f.read()
                    next_action='md'
        with col2:
            st.markdown(get_table_download_link(file), unsafe_allow_html=True)
        with col3:
            if st.button("📂", key="open_"+file):  # open emoji button
                with open(file, 'r') as f:
                    file_contents = f.read()
                    next_action='open'
        with col4:
            if st.button("🔍", key="read_"+file):  # search emoji button
                with open(file, 'r') as f:
                    file_contents = f.read()
                    next_action='search'
        with col5:
            if st.button("🗑", key="delete_"+file):
                os.remove(file)
                st.experimental_rerun()
    if len(file_contents) > 0:
        if next_action=='open':
            file_content_area = st.text_area("File Contents:", file_contents, height=500)
        if next_action=='md':
            st.markdown(file_contents)
        if next_action=='search':
            file_content_area = st.text_area("File Contents:", file_contents, height=500)
            st.write('Reasoning with your inputs...')
            response = chat_with_model(user_prompt, file_contents, model_choice)
            filename = generate_filename(file_contents, choice)
            create_file(filename, user_prompt, response, should_save)
            st.experimental_rerun()


    # Feedback
    # Step: Give User a Way to Upvote or Downvote
    feedback = st.radio("Step 8: Give your feedback", ("👍 Upvote", "👎 Downvote"))

    if feedback == "👍 Upvote":
        st.write("You upvoted 👍. Thank you for your feedback!")
    else:
        st.write("You downvoted 👎. Thank you for your feedback!")

load_dotenv()
st.write(css, unsafe_allow_html=True)
st.header("Chat with documents :books:")
user_question = st.text_input("Ask a question about your documents:")
if user_question:
    process_user_input(user_question)
with st.sidebar:
    st.subheader("Your documents")
    docs = st.file_uploader("import documents", accept_multiple_files=True)
    with st.spinner("Processing"):
        raw = pdf2txt(docs)
        if len(raw) > 0:
            length = str(len(raw))
            text_chunks = txt2chunks(raw)
            vectorstore = vector_store(text_chunks)
            st.session_state.conversation = get_chain(vectorstore)
            st.markdown('# AI Search Index of Length:' + length + ' Created.')  # add timing
            filename = generate_filename(raw, 'txt')
            create_file(filename, raw, '', should_save)

if __name__ == "__main__":
    main()
```

# AI Pair Programming - Adding a New Title and a Tip that is Factual and Interesting:

```
# Imports
import base64
import glob
import json
import math
import mistune
import openai
import os
import pytz
import re
import requests
import streamlit as st
import textract
import time
import zipfile
from audio_recorder_streamlit import audio_recorder
from bs4 import BeautifulSoup
from collections import deque
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from io import BytesIO
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from openai import ChatCompletion
from PyPDF2 import PdfReader
from templates import bot_template, css, user_template
from xml.etree import ElementTree as ET

# Constants
API_URL = 'https://qe55p8afio98s0u3.us-east-1.aws.endpoints.huggingface.cloud'  # Dr Llama
API_KEY = os.getenv('API_KEY')
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
key = os.getenv('OPENAI_API_KEY')
prompt = f"Write instructions to teach anyone to write a discharge plan. List the entities, features and relationships to CCDA and FHIR objects in boldface."
# page config and sidebar declares up front allow all other functions to see global class variables
st.set_page_config(page_title="GPT Streamlit Document Reasoner", layout="wide")

# UI Controls
should_save = st.sidebar.checkbox("💾 Save", value=True)

# Function to add witty and humor buttons
def add_witty_humor_buttons():
    with st.expander("Wit and Humor 🤣", expanded=True):
        # Tip about the Dromedary family
        st.tip("🔬 Fun Fact: Dromedaries, part of the camel family, have a single hump and are adapted to arid environments. Their 'superpowers' include the ability to survive without water for up to 7 days, thanks to their specialized blood cells and water storage in their hump.")
        
        # Define button descriptions
        descriptions = {
            "Generate Limericks 😂": "Write ten random adult limericks based on quotes that are tweet length and make you laugh 🎭",
            "Wise Quotes 🧙": "Generate ten wise quotes that are tweet length 🦉",
            "Funny Rhymes 🎤": "Create ten funny rhymes that are tweet length 🎶",
            "Medical Jokes 💉": "Create ten medical jokes that are tweet length 🏥",
            "Minnesota Humor ❄️": "Create ten jokes about Minnesota that are tweet length 🌨️",
            "Top Funny Stories 📖": "Create ten funny stories that are tweet length 📚",
            "More Funny Rhymes 🎙️": "Create ten more funny rhymes that are tweet length 🎵"
        }
        
        # Create columns
        col1, col2, col3 = st.columns([1, 1, 1], gap="small")
        
        # Add buttons to columns
        if col1.button("Generate Limericks 😂"):
            StreamLLMChatResponse(descriptions["Generate Limericks 😂"])
        
        if col2.button("Wise Quotes 🧙"):
            StreamLLMChatResponse(descriptions["Wise Quotes 🧙"])
        
        if col3.button("Funny Rhymes 🎤"):
            StreamLLMChatResponse(descriptions["Funny Rhymes 🎤"])
        
        col4, col5, col6 = st.columns([1, 1, 1], gap="small")
        
        if col4.button("Medical Jokes 💉"):
            StreamLLMChatResponse(descriptions["Medical Jokes 💉"])
        
        if col5.button("Minnesota Humor ❄️"):
            StreamLLMChatResponse(descriptions["Minnesota Humor ❄️"])
        
        if col6.button("Top Funny Stories 📖"):
            StreamLLMChatResponse(descriptions["Top Funny Stories 📖"])
        
        col7 = st.columns(1, gap="small")
        
        if col7[0].button("More Funny Rhymes 🎙️"):
            StreamLLMChatResponse(descriptions["More Funny Rhymes 🎙️"])


# Function to Stream Inference Client for Inference Endpoint Responses
def StreamLLMChatResponse(prompt):
    endpoint_url = API_URL
    hf_token = API_KEY
    client = InferenceClient(endpoint_url, token=hf_token)
    gen_kwargs = dict(
        max_new_tokens=512,
        top_k=30,
        top_p=0.9,
        temperature=0.2,
        repetition_penalty=1.02,
        stop_sequences=["\nUser:", "<|endoftext|>", "</s>"],
    )
    stream = client.text_generation(prompt, stream=True, details=True, **gen_kwargs)
    report=[]
    res_box = st.empty()
    collected_chunks=[]
    collected_messages=[]
    for r in stream:
        if r.token.special:
            continue
        if r.token.text in gen_kwargs["stop_sequences"]:
            break
        collected_chunks.append(r.token.text)
        chunk_message = r.token.text
        collected_messages.append(chunk_message)
        try:
            report.append(r.token.text)
            if len(r.token.text) > 0:
                result="".join(report).strip()
                res_box.markdown(f'*{result}*')
        except:
            st.write(' ')

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    st.markdown(response.json())
    return response.json()

def get_output(prompt):
    return query({"inputs": prompt})

def generate_filename(prompt, file_type):
    central = pytz.timezone('US/Central')
    safe_date_time = datetime.now(central).strftime("%m%d_%H%M")
    replaced_prompt = prompt.replace(" ", "_").replace("\n", "_")
    safe_prompt = "".join(x for x in replaced_prompt if x.isalnum() or x == "_")[:90]
    return f"{safe_date_time}_{safe_prompt}.{file_type}"

def transcribe_audio(openai_key, file_path, model):
    openai.api_key = openai_key
    OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {openai_key}",
    }
    with open(file_path, 'rb') as f:
        data = {'file': f}
        response = requests.post(OPENAI_API_URL, headers=headers, files=data, data={'model': model})
    if response.status_code == 200:
        st.write(response.json())
        chatResponse = chat_with_model(response.json().get('text'), '') # *************************************
        transcript = response.json().get('text')
        filename = generate_filename(transcript, 'txt')
        response = chatResponse
        user_prompt = transcript
        create_file(filename, user_prompt, response, should_save)
        return transcript
    else:
        st.write(response.json())
        st.error("Error in API call.")
        return None

def save_and_play_audio(audio_recorder):
    audio_bytes = audio_recorder()
    if audio_bytes:
        filename = generate_filename("Recording", "wav")
        with open(filename, 'wb') as f:
            f.write(audio_bytes)
        st.audio(audio_bytes, format="audio/wav")
        return filename
    return None

def create_file(filename, prompt, response, should_save=True):
    if not should_save:
        return
    base_filename, ext = os.path.splitext(filename)
    has_python_code = bool(re.search(r"```python([\s\S]*?)```", response))
    if ext in ['.txt', '.htm', '.md']:
        with open(f"{base_filename}-Prompt.txt", 'w') as file:
            file.write(prompt)
        with open(f"{base_filename}-Response.md", 'w') as file:
            file.write(response)
        if has_python_code:
            python_code = re.findall(r"```python([\s\S]*?)```", response)[0].strip()
            with open(f"{base_filename}-Code.py", 'w') as file:
                file.write(python_code)
            
def truncate_document(document, length):
    return document[:length]

def divide_document(document, max_length):
    return [document[i:i+max_length] for i in range(0, len(document), max_length)]

def get_table_download_link(file_path):
    with open(file_path, 'r') as file:
        try:
            data = file.read()
        except:
            st.write('')
            return file_path    
    b64 = base64.b64encode(data.encode()).decode()  
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1]  # get the file extension
    if ext == '.txt':
        mime_type = 'text/plain'
    elif ext == '.py':
        mime_type = 'text/plain'
    elif ext == '.xlsx':
        mime_type = 'text/plain'
    elif ext == '.csv':
        mime_type = 'text/plain'
    elif ext == '.htm':
        mime_type = 'text/html'
    elif ext == '.md':
        mime_type = 'text/markdown'
    else:
        mime_type = 'application/octet-stream'  # general binary data type
    href = f'<a href="data:{mime_type};base64,{b64}" target="_blank" download="{file_name}">{file_name}</a>'
    return href

def CompressXML(xml_text):
    root = ET.fromstring(xml_text)
    for elem in list(root.iter()):
        if isinstance(elem.tag, str) and 'Comment' in elem.tag:
            elem.parent.remove(elem)
    return ET.tostring(root, encoding='unicode', method="xml")
    
def read_file_content(file,max_length):
    if file.type == "application/json":
        content = json.load(file)
        return str(content)
    elif file.type == "text/html" or file.type == "text/htm":
        content = BeautifulSoup(file, "html.parser")
        return content.text
    elif file.type == "application/xml" or file.type == "text/xml":
        tree = ET.parse(file)
        root = tree.getroot()
        xml = CompressXML(ET.tostring(root, encoding='unicode'))
        return xml
    elif file.type == "text/markdown" or file.type == "text/md":
        md = mistune.create_markdown()
        content = md(file.read().decode())
        return content
    elif file.type == "text/plain":
        return file.getvalue().decode()
    else:
        return ""

def chat_with_model(prompt, document_section, model_choice='gpt-3.5-turbo'):
    model = model_choice
    conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    conversation.append({'role': 'user', 'content': prompt})
    if len(document_section)>0:
        conversation.append({'role': 'assistant', 'content': document_section})
    start_time = time.time()
    report = []
    res_box = st.empty()
    collected_chunks = []
    collected_messages = []
    for chunk in openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=conversation, temperature=0.5, stream=True):
        collected_chunks.append(chunk)  
        chunk_message = chunk['choices'][0]['delta']  
        collected_messages.append(chunk_message) 
        content=chunk["choices"][0].get("delta",{}).get("content")
        try:
            report.append(content)
            if len(content) > 0:
                result = "".join(report).strip()
                res_box.markdown(f'*{result}*') 
        except:
            st.write(' ')
    full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
    st.write("Elapsed time:")
    st.write(time.time() - start_time)
    return full_reply_content

def chat_with_file_contents(prompt, file_content, model_choice='gpt-3.5-turbo'):
    conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    conversation.append({'role': 'user', 'content': prompt})
    if len(file_content)>0:
        conversation.append({'role': 'assistant', 'content': file_content})
    response = openai.ChatCompletion.create(model=model_choice, messages=conversation)
    return response['choices'][0]['message']['content']

def extract_mime_type(file):
    if isinstance(file, str):
        pattern = r"type='(.*?)'"
        match = re.search(pattern, file)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Unable to extract MIME type from {file}")
    elif isinstance(file, streamlit.UploadedFile):
        return file.type
    else:
        raise TypeError("Input should be a string or a streamlit.UploadedFile object")

def extract_file_extension(file):
    # get the file name directly from the UploadedFile object
    file_name = file.name
    pattern = r".*?\.(.*?)$"
    match = re.search(pattern, file_name)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Unable to extract file extension from {file_name}")

def pdf2txt(docs):
    text = ""
    for file in docs:
        file_extension = extract_file_extension(file)
        st.write(f"File type extension: {file_extension}")
        try:
            if file_extension.lower() in ['py', 'txt', 'html', 'htm', 'xml', 'json']:
                text += file.getvalue().decode('utf-8')
            elif file_extension.lower() == 'pdf':
                from PyPDF2 import PdfReader
                pdf = PdfReader(BytesIO(file.getvalue()))
                for page in range(len(pdf.pages)):
                    text += pdf.pages[page].extract_text() # new PyPDF2 syntax
        except Exception as e:
            st.write(f"Error processing file {file.name}: {e}")
    return text

def txt2chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

def vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=key)
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

def process_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        filename = generate_filename(user_question, 'txt')
        response = message.content
        user_prompt = user_question
        create_file(filename, user_prompt, response, should_save)       

def divide_prompt(prompt, max_length):
    words = prompt.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        if len(word) + current_length <= max_length:
            current_length += len(word) + 1 
            current_chunk.append(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
    chunks.append(' '.join(current_chunk))
    return chunks

def create_zip_of_files(files):
    zip_name = "all_files.zip"
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for file in files:
            zipf.write(file)
    return zip_name

def get_zip_download_link(zip_file):
    with open(zip_file, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/zip;base64,{b64}" download="{zip_file}">Download All</a>'
    return href

def main():

    st.title("DromeLlama7B")
    prompt = f"Write ten funny jokes that are tweet length stories that make you laugh.  Show as markdown outline with emojis for each."

    # Add Wit and Humor buttons
    add_witty_humor_buttons()
    
    example_input = st.text_input("Enter your example text:", value=prompt)
    if st.button("Run Prompt With DromeLlama"):
        try:
            StreamLLMChatResponse(example_input)
        except:
            st.write('DromeLlama is asleep.  Starting up now on A10 - please give 5 minutes then retry as KEDA scales up from zero to activate running container(s).')

    
    openai.api_key = os.getenv('OPENAI_KEY')
    menu = ["txt", "htm", "xlsx", "csv", "md", "py"]
    choice = st.sidebar.selectbox("Output File Type:", menu)
    model_choice = st.sidebar.radio("Select Model:", ('gpt-3.5-turbo', 'gpt-3.5-turbo-0301'))
    filename = save_and_play_audio(audio_recorder)
    if filename is not None:
        transcription = transcribe_audio(key, filename, "whisper-1")
        st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)
        filename = None
    user_prompt = st.text_area("Enter prompts, instructions & questions:", '', height=100)
    collength, colupload = st.columns([2,3])  # adjust the ratio as needed
    with collength:
        max_length = st.slider("File section length for large files", min_value=1000, max_value=128000, value=12000, step=1000)
    with colupload:
        uploaded_file = st.file_uploader("Add a file for context:", type=["pdf", "xml", "json", "xlsx", "csv", "html", "htm", "md", "txt"])
    document_sections = deque()
    document_responses = {}
    if uploaded_file is not None:
        file_content = read_file_content(uploaded_file, max_length)
        document_sections.extend(divide_document(file_content, max_length))
    if len(document_sections) > 0:
        if st.button("👁️ View Upload"):
            st.markdown("**Sections of the uploaded file:**")
            for i, section in enumerate(list(document_sections)):
                st.markdown(f"**Section {i+1}**\n{section}")
        st.markdown("**Chat with the model:**")
        for i, section in enumerate(list(document_sections)):
            if i in document_responses:
                st.markdown(f"**Section {i+1}**\n{document_responses[i]}")
            else:
                if st.button(f"Chat about Section {i+1}"):
                    st.write('Reasoning with your inputs...')
                    response = chat_with_model(user_prompt, section, model_choice)
                    st.write('Response:')
                    st.write(response)
                    document_responses[i] = response
                    filename = generate_filename(f"{user_prompt}_section_{i+1}", choice)
                    create_file(filename, user_prompt, response, should_save)
                    st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)
    if st.button('💬 Chat'):
        st.write('Reasoning with your inputs...')
        user_prompt_sections = divide_prompt(user_prompt, max_length)
        full_response = ''
        for prompt_section in user_prompt_sections:
            response = chat_with_model(prompt_section, ''.join(list(document_sections)), model_choice)
            full_response += response + '\n'  # Combine the responses
        response = full_response
        st.write('Response:')
        st.write(response)
        filename = generate_filename(user_prompt, choice)
        create_file(filename, user_prompt, response, should_save)
        st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)
    all_files = glob.glob("*.*")
    all_files = [file for file in all_files if len(os.path.splitext(file)[0]) >= 20]  # exclude files with short names
    all_files.sort(key=lambda x: (os.path.splitext(x)[1], x), reverse=True)  # sort by file type and file name in descending order
    if st.sidebar.button("🗑 Delete All"):
        for file in all_files:
            os.remove(file)
        st.experimental_rerun()
    if st.sidebar.button("⬇️ Download All"):
        zip_file = create_zip_of_files(all_files)
        st.sidebar.markdown(get_zip_download_link(zip_file), unsafe_allow_html=True)
    file_contents=''
    next_action=''
    for file in all_files:
        col1, col2, col3, col4, col5 = st.sidebar.columns([1,6,1,1,1])  # adjust the ratio as needed
        with col1:
            if st.button("🌐", key="md_"+file):  # md emoji button
                with open(file, 'r') as f:
                    file_contents = f.read()
                    next_action='md'
        with col2:
            st.markdown(get_table_download_link(file), unsafe_allow_html=True)
        with col3:
            if st.button("📂", key="open_"+file):  # open emoji button
                with open(file, 'r') as f:
                    file_contents = f.read()
                    next_action='open'
        with col4:
            if st.button("🔍", key="read_"+file):  # search emoji button
                with open(file, 'r') as f:
                    file_contents = f.read()
                    next_action='search'
        with col5:
            if st.button("🗑", key="delete_"+file):
                os.remove(file)
                st.experimental_rerun()
    if len(file_contents) > 0:
        if next_action=='open':
            file_content_area = st.text_area("File Contents:", file_contents, height=500)
        if next_action=='md':
            st.markdown(file_contents)
        if next_action=='search':
            file_content_area = st.text_area("File Contents:", file_contents, height=500)
            st.write('Reasoning with your inputs...')
            response = chat_with_model(user_prompt, file_contents, model_choice)
            filename = generate_filename(file_contents, choice)
            create_file(filename, user_prompt, response, should_save)
            st.experimental_rerun()


    # Feedback
    # Step: Give User a Way to Upvote or Downvote
    feedback = st.radio("Step 8: Give your feedback", ("👍 Upvote", "👎 Downvote"))

    if feedback == "👍 Upvote":
        st.write("You upvoted 👍. Thank you for your feedback!")
    else:
        st.write("You downvoted 👎. Thank you for your feedback!")

load_dotenv()
st.write(css, unsafe_allow_html=True)
st.header("Chat with documents :books:")
user_question = st.text_input("Ask a question about your documents:")
if user_question:
    process_user_input(user_question)
with st.sidebar:
    st.subheader("Your documents")
    docs = st.file_uploader("import documents", accept_multiple_files=True)
    with st.spinner("Processing"):
        raw = pdf2txt(docs)
        if len(raw) > 0:
            length = str(len(raw))
            text_chunks = txt2chunks(raw)
            vectorstore = vector_store(text_chunks)
            st.session_state.conversation = get_chain(vectorstore)
            st.markdown('# AI Search Index of Length:' + length + ' Created.')  # add timing
            filename = generate_filename(raw, 'txt')
            create_file(filename, raw, '', should_save)

if __name__ == "__main__":
    main()
```

# AI Pair Programming - Programmatic UI Layout
```
use st.columns to make the buttons side by side in layout with small gap.  Use specification below :  Function signature[source]
st.columns(spec, *, gap="small")

Parameters
spec (int or iterable of numbers)

Controls the number and width of columns to insert. Can be one of:

An integer that specifies the number of columns. All columns have equal width in this case.
An iterable of numbers (int or float) that specify the relative width of each column. E.g. [0.7, 0.3] creates two columns where the first one takes up 70% of the available with and the second one takes up 30%. Or [1, 2, 3] creates three columns where the second one is two times the width of the first one, and the third one is three times that width.
gap ("small", "medium", or "large")

The size of the gap between the columns. Defaults to "small". This argument can only be supplied by keyword.

Returns
(list of containers)

A list of container objects.
```

# AI Pair Programming - AI Pipeline Targetd Refactor to Infer Generality
```
Modify this function to add two more buttons with a key and label that is unique.  Add to existing one.  Create one with wise quotes, one with funny rhymes, one with medical jokes, one with minnesota humor, one with top funny stories, and one with funny rhymes.  # Function to add witty and humor buttons
def add_witty_humor_buttons():
    with st.expander("Wit and Humor 🤣", expanded=True):
        button_description = "Write ten random adult limericks based on quotes that are tweet length and make you laugh 🎭"
        button_label = "Generate Limericks 😂"
        if st.button(button_label):
            try:
                StreamLLMChatResponse(button_description)
            except:
                st.write('Dr. Llama is asleep.  Starting up now on A10 - please give 5 minutes then retry as KEDA scales up from zero to activate running container(s).')
```




# AI Pipeline - Wit Courtesy of Llama 7B:
```
# Imports
import base64
import glob
import json
import math
import mistune
import openai
import os
import pytz
import re
import requests
import streamlit as st
import textract
import time
import zipfile
from audio_recorder_streamlit import audio_recorder
from bs4 import BeautifulSoup
from collections import deque
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from io import BytesIO
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from openai import ChatCompletion
from PyPDF2 import PdfReader
from templates import bot_template, css, user_template
from xml.etree import ElementTree as ET

# Constants
API_URL = 'https://qe55p8afio98s0u3.us-east-1.aws.endpoints.huggingface.cloud'  # Dr Llama
API_KEY = os.getenv('API_KEY')
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
key = os.getenv('OPENAI_API_KEY')
prompt = f"Write instructions to teach anyone to write a discharge plan. List the entities, features and relationships to CCDA and FHIR objects in boldface."
# page config and sidebar declares up front allow all other functions to see global class variables
st.set_page_config(page_title="GPT Streamlit Document Reasoner", layout="wide")

# UI Controls
should_save = st.sidebar.checkbox("💾 Save", value=True)

# Functions

# Function to add witty and humor buttons
def add_witty_humor_buttons():
    with st.expander("Wit and Humor 🤣", expanded=True):
        button_description = "Write ten random adult limericks based on quotes that are tweet length and make you laugh 🎭"
        button_label = "Generate Limericks 😂"
        
        if st.button(button_label):
            StreamLLMChatResponse(button_description)

# Function to Stream Inference Client for Inference Endpoint Responses
def StreamLLMChatResponse(prompt):
    endpoint_url = API_URL
    hf_token = API_KEY
    client = InferenceClient(endpoint_url, token=hf_token)
    gen_kwargs = dict(
        max_new_tokens=512,
        top_k=30,
        top_p=0.9,
        temperature=0.2,
        repetition_penalty=1.02,
        stop_sequences=["\nUser:", "<|endoftext|>", "</s>"],
    )
    stream = client.text_generation(prompt, stream=True, details=True, **gen_kwargs)
    report=[]
    res_box = st.empty()
    collected_chunks=[]
    collected_messages=[]
    for r in stream:
        if r.token.special:
            continue
        if r.token.text in gen_kwargs["stop_sequences"]:
            break
        collected_chunks.append(r.token.text)
        chunk_message = r.token.text
        collected_messages.append(chunk_message)
        try:
            report.append(r.token.text)
            if len(r.token.text) > 0:
                result="".join(report).strip()
                res_box.markdown(f'*{result}*')
        except:
            st.write(' ')

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    st.markdown(response.json())
    return response.json()

def get_output(prompt):
    return query({"inputs": prompt})

def generate_filename(prompt, file_type):
    central = pytz.timezone('US/Central')
    safe_date_time = datetime.now(central).strftime("%m%d_%H%M")
    replaced_prompt = prompt.replace(" ", "_").replace("\n", "_")
    safe_prompt = "".join(x for x in replaced_prompt if x.isalnum() or x == "_")[:90]
    return f"{safe_date_time}_{safe_prompt}.{file_type}"

def transcribe_audio(openai_key, file_path, model):
    openai.api_key = openai_key
    OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {openai_key}",
    }
    with open(file_path, 'rb') as f:
        data = {'file': f}
        response = requests.post(OPENAI_API_URL, headers=headers, files=data, data={'model': model})
    if response.status_code == 200:
        st.write(response.json())
        chatResponse = chat_with_model(response.json().get('text'), '') # *************************************
        transcript = response.json().get('text')
        filename = generate_filename(transcript, 'txt')
        response = chatResponse
        user_prompt = transcript
        create_file(filename, user_prompt, response, should_save)
        return transcript
    else:
        st.write(response.json())
        st.error("Error in API call.")
        return None

def save_and_play_audio(audio_recorder):
    audio_bytes = audio_recorder()
    if audio_bytes:
        filename = generate_filename("Recording", "wav")
        with open(filename, 'wb') as f:
            f.write(audio_bytes)
        st.audio(audio_bytes, format="audio/wav")
        return filename
    return None

def create_file(filename, prompt, response, should_save=True):
    if not should_save:
        return
    base_filename, ext = os.path.splitext(filename)
    has_python_code = bool(re.search(r"```python([\s\S]*?)```", response))
    if ext in ['.txt', '.htm', '.md']:
        with open(f"{base_filename}-Prompt.txt", 'w') as file:
            file.write(prompt)
        with open(f"{base_filename}-Response.md", 'w') as file:
            file.write(response)
        if has_python_code:
            python_code = re.findall(r"```python([\s\S]*?)```", response)[0].strip()
            with open(f"{base_filename}-Code.py", 'w') as file:
                file.write(python_code)
            
def truncate_document(document, length):
    return document[:length]

def divide_document(document, max_length):
    return [document[i:i+max_length] for i in range(0, len(document), max_length)]

def get_table_download_link(file_path):
    with open(file_path, 'r') as file:
        try:
            data = file.read()
        except:
            st.write('')
            return file_path    
    b64 = base64.b64encode(data.encode()).decode()  
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1]  # get the file extension
    if ext == '.txt':
        mime_type = 'text/plain'
    elif ext == '.py':
        mime_type = 'text/plain'
    elif ext == '.xlsx':
        mime_type = 'text/plain'
    elif ext == '.csv':
        mime_type = 'text/plain'
    elif ext == '.htm':
        mime_type = 'text/html'
    elif ext == '.md':
        mime_type = 'text/markdown'
    else:
        mime_type = 'application/octet-stream'  # general binary data type
    href = f'<a href="data:{mime_type};base64,{b64}" target="_blank" download="{file_name}">{file_name}</a>'
    return href

def CompressXML(xml_text):
    root = ET.fromstring(xml_text)
    for elem in list(root.iter()):
        if isinstance(elem.tag, str) and 'Comment' in elem.tag:
            elem.parent.remove(elem)
    return ET.tostring(root, encoding='unicode', method="xml")
    
def read_file_content(file,max_length):
    if file.type == "application/json":
        content = json.load(file)
        return str(content)
    elif file.type == "text/html" or file.type == "text/htm":
        content = BeautifulSoup(file, "html.parser")
        return content.text
    elif file.type == "application/xml" or file.type == "text/xml":
        tree = ET.parse(file)
        root = tree.getroot()
        xml = CompressXML(ET.tostring(root, encoding='unicode'))
        return xml
    elif file.type == "text/markdown" or file.type == "text/md":
        md = mistune.create_markdown()
        content = md(file.read().decode())
        return content
    elif file.type == "text/plain":
        return file.getvalue().decode()
    else:
        return ""

def chat_with_model(prompt, document_section, model_choice='gpt-3.5-turbo'):
    model = model_choice
    conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    conversation.append({'role': 'user', 'content': prompt})
    if len(document_section)>0:
        conversation.append({'role': 'assistant', 'content': document_section})
    start_time = time.time()
    report = []
    res_box = st.empty()
    collected_chunks = []
    collected_messages = []
    for chunk in openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=conversation, temperature=0.5, stream=True):
        collected_chunks.append(chunk)  
        chunk_message = chunk['choices'][0]['delta']  
        collected_messages.append(chunk_message) 
        content=chunk["choices"][0].get("delta",{}).get("content")
        try:
            report.append(content)
            if len(content) > 0:
                result = "".join(report).strip()
                res_box.markdown(f'*{result}*') 
        except:
            st.write(' ')
    full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
    st.write("Elapsed time:")
    st.write(time.time() - start_time)
    return full_reply_content

def chat_with_file_contents(prompt, file_content, model_choice='gpt-3.5-turbo'):
    conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    conversation.append({'role': 'user', 'content': prompt})
    if len(file_content)>0:
        conversation.append({'role': 'assistant', 'content': file_content})
    response = openai.ChatCompletion.create(model=model_choice, messages=conversation)
    return response['choices'][0]['message']['content']

def extract_mime_type(file):
    if isinstance(file, str):
        pattern = r"type='(.*?)'"
        match = re.search(pattern, file)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Unable to extract MIME type from {file}")
    elif isinstance(file, streamlit.UploadedFile):
        return file.type
    else:
        raise TypeError("Input should be a string or a streamlit.UploadedFile object")

def extract_file_extension(file):
    # get the file name directly from the UploadedFile object
    file_name = file.name
    pattern = r".*?\.(.*?)$"
    match = re.search(pattern, file_name)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Unable to extract file extension from {file_name}")

def pdf2txt(docs):
    text = ""
    for file in docs:
        file_extension = extract_file_extension(file)
        st.write(f"File type extension: {file_extension}")
        try:
            if file_extension.lower() in ['py', 'txt', 'html', 'htm', 'xml', 'json']:
                text += file.getvalue().decode('utf-8')
            elif file_extension.lower() == 'pdf':
                from PyPDF2 import PdfReader
                pdf = PdfReader(BytesIO(file.getvalue()))
                for page in range(len(pdf.pages)):
                    text += pdf.pages[page].extract_text() # new PyPDF2 syntax
        except Exception as e:
            st.write(f"Error processing file {file.name}: {e}")
    return text

def txt2chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

def vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=key)
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

def process_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        filename = generate_filename(user_question, 'txt')
        response = message.content
        user_prompt = user_question
        create_file(filename, user_prompt, response, should_save)       

def divide_prompt(prompt, max_length):
    words = prompt.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        if len(word) + current_length <= max_length:
            current_length += len(word) + 1 
            current_chunk.append(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
    chunks.append(' '.join(current_chunk))
    return chunks

def create_zip_of_files(files):
    zip_name = "all_files.zip"
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for file in files:
            zipf.write(file)
    return zip_name

def get_zip_download_link(zip_file):
    with open(zip_file, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/zip;base64,{b64}" download="{zip_file}">Download All</a>'
    return href

def main():


    st.title(" DrLlama7B")
    prompt = f"Write ten funny jokes that are tweet length stories that make you laugh.  Show as markdown outline with emojis for each."
    
    # Add Wit and Humor buttons
    add_witty_humor_buttons()
    
    example_input = st.text_input("Enter your example text:", value=prompt)
    if st.button("Run Prompt With Dr Llama"):
        try:
            StreamLLMChatResponse(example_input)
        except:
            st.write('Dr. Llama is asleep.  Starting up now on A10 - please give 5 minutes then retry as KEDA scales up from zero to activate running container(s).')

    
    openai.api_key = os.getenv('OPENAI_KEY')
    menu = ["txt", "htm", "xlsx", "csv", "md", "py"]
    choice = st.sidebar.selectbox("Output File Type:", menu)
    model_choice = st.sidebar.radio("Select Model:", ('gpt-3.5-turbo', 'gpt-3.5-turbo-0301'))
    filename = save_and_play_audio(audio_recorder)
    if filename is not None:
        transcription = transcribe_audio(key, filename, "whisper-1")
        st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)
        filename = None
    user_prompt = st.text_area("Enter prompts, instructions & questions:", '', height=100)
    collength, colupload = st.columns([2,3])  # adjust the ratio as needed
    with collength:
        max_length = st.slider("File section length for large files", min_value=1000, max_value=128000, value=12000, step=1000)
    with colupload:
        uploaded_file = st.file_uploader("Add a file for context:", type=["pdf", "xml", "json", "xlsx", "csv", "html", "htm", "md", "txt"])
    document_sections = deque()
    document_responses = {}
    if uploaded_file is not None:
        file_content = read_file_content(uploaded_file, max_length)
        document_sections.extend(divide_document(file_content, max_length))
    if len(document_sections) > 0:
        if st.button("👁️ View Upload"):
            st.markdown("**Sections of the uploaded file:**")
            for i, section in enumerate(list(document_sections)):
                st.markdown(f"**Section {i+1}**\n{section}")
        st.markdown("**Chat with the model:**")
        for i, section in enumerate(list(document_sections)):
            if i in document_responses:
                st.markdown(f"**Section {i+1}**\n{document_responses[i]}")
            else:
                if st.button(f"Chat about Section {i+1}"):
                    st.write('Reasoning with your inputs...')
                    response = chat_with_model(user_prompt, section, model_choice)
                    st.write('Response:')
                    st.write(response)
                    document_responses[i] = response
                    filename = generate_filename(f"{user_prompt}_section_{i+1}", choice)
                    create_file(filename, user_prompt, response, should_save)
                    st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)
    if st.button('💬 Chat'):
        st.write('Reasoning with your inputs...')
        user_prompt_sections = divide_prompt(user_prompt, max_length)
        full_response = ''
        for prompt_section in user_prompt_sections:
            response = chat_with_model(prompt_section, ''.join(list(document_sections)), model_choice)
            full_response += response + '\n'  # Combine the responses
        response = full_response
        st.write('Response:')
        st.write(response)
        filename = generate_filename(user_prompt, choice)
        create_file(filename, user_prompt, response, should_save)
        st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)
    all_files = glob.glob("*.*")
    all_files = [file for file in all_files if len(os.path.splitext(file)[0]) >= 20]  # exclude files with short names
    all_files.sort(key=lambda x: (os.path.splitext(x)[1], x), reverse=True)  # sort by file type and file name in descending order
    if st.sidebar.button("🗑 Delete All"):
        for file in all_files:
            os.remove(file)
        st.experimental_rerun()
    if st.sidebar.button("⬇️ Download All"):
        zip_file = create_zip_of_files(all_files)
        st.sidebar.markdown(get_zip_download_link(zip_file), unsafe_allow_html=True)
    file_contents=''
    next_action=''
    for file in all_files:
        col1, col2, col3, col4, col5 = st.sidebar.columns([1,6,1,1,1])  # adjust the ratio as needed
        with col1:
            if st.button("🌐", key="md_"+file):  # md emoji button
                with open(file, 'r') as f:
                    file_contents = f.read()
                    next_action='md'
        with col2:
            st.markdown(get_table_download_link(file), unsafe_allow_html=True)
        with col3:
            if st.button("📂", key="open_"+file):  # open emoji button
                with open(file, 'r') as f:
                    file_contents = f.read()
                    next_action='open'
        with col4:
            if st.button("🔍", key="read_"+file):  # search emoji button
                with open(file, 'r') as f:
                    file_contents = f.read()
                    next_action='search'
        with col5:
            if st.button("🗑", key="delete_"+file):
                os.remove(file)
                st.experimental_rerun()
    if len(file_contents) > 0:
        if next_action=='open':
            file_content_area = st.text_area("File Contents:", file_contents, height=500)
        if next_action=='md':
            st.markdown(file_contents)
        if next_action=='search':
            file_content_area = st.text_area("File Contents:", file_contents, height=500)
            st.write('Reasoning with your inputs...')
            response = chat_with_model(user_prompt, file_contents, model_choice)
            filename = generate_filename(file_contents, choice)
            create_file(filename, user_prompt, response, should_save)
            st.experimental_rerun()


    # Feedback
    # Step: Give User a Way to Upvote or Downvote
    feedback = st.radio("Step 8: Give your feedback", ("👍 Upvote", "👎 Downvote"))

    if feedback == "👍 Upvote":
        st.write("You upvoted 👍. Thank you for your feedback!")
    else:
        st.write("You downvoted 👎. Thank you for your feedback!")

load_dotenv()
st.write(css, unsafe_allow_html=True)
st.header("Chat with documents :books:")
user_question = st.text_input("Ask a question about your documents:")
if user_question:
    process_user_input(user_question)
with st.sidebar:
    st.subheader("Your documents")
    docs = st.file_uploader("import documents", accept_multiple_files=True)
    with st.spinner("Processing"):
        raw = pdf2txt(docs)
        if len(raw) > 0:
            length = str(len(raw))
            text_chunks = txt2chunks(raw)
            vectorstore = vector_store(text_chunks)
            st.session_state.conversation = get_chain(vectorstore)
            st.markdown('# AI Search Index of Length:' + length + ' Created.')  # add timing
            filename = generate_filename(raw, 'txt')
            create_file(filename, raw, '', should_save)

if __name__ == "__main__":
    main()

```



# Transcript input from YouTube - Content Training for Tasks and Opportunities:
# a. Youtube Playlist with Videos and Transcripts for Base Knowledge:
1. School of AI - Favorite Teachers : https://www.youtube.com/playlist?list=PLHgX2IExbFot_VUzRyr3Y70ApBNIPmU7U  ,  https://www.youtube.com/watch?v=kyY9PSQRH1Q&list=PLHgX2IExbFot_VUzRyr3Y70ApBNIPmU7U

# AI Pipeline - Joke and Quote writing for maximum readability
```
Modify the program below to add a few picture buttons which will use emojis and a witty title to describe a prompt.  the first prompt I want is "Write ten random adult limerick based on quotes that are tweet length and make you laugh.  Show as numbered bold faced and large font markdown outline with emojis for each."  Modify this code to add the prompt emoji labeled buttons above the text box.  when you click them pass the varible they contain to a function which runs the chat through the Llama web service call in the code below.  refactor it so it is function based.  Put variables that set description for button and label for button right before the st.button() function calls and use st.expander() function to create a expanded description container with a witty label so user could collapse st.expander to hide buttons of a particular type.  This first type will be Wit and Humor.  Make sure each label contains appropriate emojis.  Code:  # Imports
import base64
import glob
import json
import math
import mistune
import openai
import os
import pytz
import re
import requests
import streamlit as st
import textract
import time
import zipfile
from audio_recorder_streamlit import audio_recorder
from bs4 import BeautifulSoup
from collections import deque
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from io import BytesIO
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from openai import ChatCompletion
from PyPDF2 import PdfReader
from templates import bot_template, css, user_template
from xml.etree import ElementTree as ET

# Constants
API_URL = 'https://qe55p8afio98s0u3.us-east-1.aws.endpoints.huggingface.cloud'  # Dr Llama
API_KEY = os.getenv('API_KEY')
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
key = os.getenv('OPENAI_API_KEY')
prompt = f"Write instructions to teach anyone to write a discharge plan. List the entities, features and relationships to CCDA and FHIR objects in boldface."
# page config and sidebar declares up front allow all other functions to see global class variables
st.set_page_config(page_title="GPT Streamlit Document Reasoner", layout="wide")

# UI Controls
should_save = st.sidebar.checkbox("💾 Save", value=True)

# Functions
def StreamLLMChatResponse(prompt):
    endpoint_url = API_URL
    hf_token = API_KEY
    client = InferenceClient(endpoint_url, token=hf_token)
    gen_kwargs = dict(
        max_new_tokens=512,
        top_k=30,
        top_p=0.9,
        temperature=0.2,
        repetition_penalty=1.02,
        stop_sequences=["\nUser:", "<|endoftext|>", "</s>"],
    )
    stream = client.text_generation(prompt, stream=True, details=True, **gen_kwargs)
    report=[]
    res_box = st.empty()
    collected_chunks=[]
    collected_messages=[]
    for r in stream:
        if r.token.special:
            continue
        if r.token.text in gen_kwargs["stop_sequences"]:
            break
        collected_chunks.append(r.token.text)
        chunk_message = r.token.text
        collected_messages.append(chunk_message)
        try:
            report.append(r.token.text)
            if len(r.token.text) > 0:
                result="".join(report).strip()
                res_box.markdown(f'*{result}*')
        except:
            st.write(' ')

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    st.markdown(response.json())
    return response.json()

def get_output(prompt):
    return query({"inputs": prompt})

def generate_filename(prompt, file_type):
    central = pytz.timezone('US/Central')
    safe_date_time = datetime.now(central).strftime("%m%d_%H%M")
    replaced_prompt = prompt.replace(" ", "_").replace("\n", "_")
    safe_prompt = "".join(x for x in replaced_prompt if x.isalnum() or x == "_")[:90]
    return f"{safe_date_time}_{safe_prompt}.{file_type}"

def transcribe_audio(openai_key, file_path, model):
    openai.api_key = openai_key
    OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {openai_key}",
    }
    with open(file_path, 'rb') as f:
        data = {'file': f}
        response = requests.post(OPENAI_API_URL, headers=headers, files=data, data={'model': model})
    if response.status_code == 200:
        st.write(response.json())
        chatResponse = chat_with_model(response.json().get('text'), '') # *************************************
        transcript = response.json().get('text')
        filename = generate_filename(transcript, 'txt')
        response = chatResponse
        user_prompt = transcript
        create_file(filename, user_prompt, response, should_save)
        return transcript
    else:
        st.write(response.json())
        st.error("Error in API call.")
        return None

def save_and_play_audio(audio_recorder):
    audio_bytes = audio_recorder()
    if audio_bytes:
        filename = generate_filename("Recording", "wav")
        with open(filename, 'wb') as f:
            f.write(audio_bytes)
        st.audio(audio_bytes, format="audio/wav")
        return filename
    return None

def create_file(filename, prompt, response, should_save=True):
    if not should_save:
        return
    base_filename, ext = os.path.splitext(filename)
    has_python_code = bool(re.search(r"```python([\s\S]*?)```", response))
    if ext in ['.txt', '.htm', '.md']:
        with open(f"{base_filename}-Prompt.txt", 'w') as file:
            file.write(prompt)
        with open(f"{base_filename}-Response.md", 'w') as file:
            file.write(response)
        if has_python_code:
            python_code = re.findall(r"```python([\s\S]*?)```", response)[0].strip()
            with open(f"{base_filename}-Code.py", 'w') as file:
                file.write(python_code)
            
def truncate_document(document, length):
    return document[:length]

def divide_document(document, max_length):
    return [document[i:i+max_length] for i in range(0, len(document), max_length)]

def get_table_download_link(file_path):
    with open(file_path, 'r') as file:
        try:
            data = file.read()
        except:
            st.write('')
            return file_path    
    b64 = base64.b64encode(data.encode()).decode()  
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1]  # get the file extension
    if ext == '.txt':
        mime_type = 'text/plain'
    elif ext == '.py':
        mime_type = 'text/plain'
    elif ext == '.xlsx':
        mime_type = 'text/plain'
    elif ext == '.csv':
        mime_type = 'text/plain'
    elif ext == '.htm':
        mime_type = 'text/html'
    elif ext == '.md':
        mime_type = 'text/markdown'
    else:
        mime_type = 'application/octet-stream'  # general binary data type
    href = f'<a href="data:{mime_type};base64,{b64}" target="_blank" download="{file_name}">{file_name}</a>'
    return href

def CompressXML(xml_text):
    root = ET.fromstring(xml_text)
    for elem in list(root.iter()):
        if isinstance(elem.tag, str) and 'Comment' in elem.tag:
            elem.parent.remove(elem)
    return ET.tostring(root, encoding='unicode', method="xml")
    
def read_file_content(file,max_length):
    if file.type == "application/json":
        content = json.load(file)
        return str(content)
    elif file.type == "text/html" or file.type == "text/htm":
        content = BeautifulSoup(file, "html.parser")
        return content.text
    elif file.type == "application/xml" or file.type == "text/xml":
        tree = ET.parse(file)
        root = tree.getroot()
        xml = CompressXML(ET.tostring(root, encoding='unicode'))
        return xml
    elif file.type == "text/markdown" or file.type == "text/md":
        md = mistune.create_markdown()
        content = md(file.read().decode())
        return content
    elif file.type == "text/plain":
        return file.getvalue().decode()
    else:
        return ""

def chat_with_model(prompt, document_section, model_choice='gpt-3.5-turbo'):
    model = model_choice
    conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    conversation.append({'role': 'user', 'content': prompt})
    if len(document_section)>0:
        conversation.append({'role': 'assistant', 'content': document_section})
    start_time = time.time()
    report = []
    res_box = st.empty()
    collected_chunks = []
    collected_messages = []
    for chunk in openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=conversation, temperature=0.5, stream=True):
        collected_chunks.append(chunk)  
        chunk_message = chunk['choices'][0]['delta']  
        collected_messages.append(chunk_message) 
        content=chunk["choices"][0].get("delta",{}).get("content")
        try:
            report.append(content)
            if len(content) > 0:
                result = "".join(report).strip()
                res_box.markdown(f'*{result}*') 
        except:
            st.write(' ')
    full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
    st.write("Elapsed time:")
    st.write(time.time() - start_time)
    return full_reply_content

def chat_with_file_contents(prompt, file_content, model_choice='gpt-3.5-turbo'):
    conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    conversation.append({'role': 'user', 'content': prompt})
    if len(file_content)>0:
        conversation.append({'role': 'assistant', 'content': file_content})
    response = openai.ChatCompletion.create(model=model_choice, messages=conversation)
    return response['choices'][0]['message']['content']

def extract_mime_type(file):
    if isinstance(file, str):
        pattern = r"type='(.*?)'"
        match = re.search(pattern, file)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Unable to extract MIME type from {file}")
    elif isinstance(file, streamlit.UploadedFile):
        return file.type
    else:
        raise TypeError("Input should be a string or a streamlit.UploadedFile object")

def extract_file_extension(file):
    # get the file name directly from the UploadedFile object
    file_name = file.name
    pattern = r".*?\.(.*?)$"
    match = re.search(pattern, file_name)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Unable to extract file extension from {file_name}")

def pdf2txt(docs):
    text = ""
    for file in docs:
        file_extension = extract_file_extension(file)
        st.write(f"File type extension: {file_extension}")
        try:
            if file_extension.lower() in ['py', 'txt', 'html', 'htm', 'xml', 'json']:
                text += file.getvalue().decode('utf-8')
            elif file_extension.lower() == 'pdf':
                from PyPDF2 import PdfReader
                pdf = PdfReader(BytesIO(file.getvalue()))
                for page in range(len(pdf.pages)):
                    text += pdf.pages[page].extract_text() # new PyPDF2 syntax
        except Exception as e:
            st.write(f"Error processing file {file.name}: {e}")
    return text

def txt2chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

def vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=key)
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

def process_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        filename = generate_filename(user_question, 'txt')
        response = message.content
        user_prompt = user_question
        create_file(filename, user_prompt, response, should_save)       

def divide_prompt(prompt, max_length):
    words = prompt.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        if len(word) + current_length <= max_length:
            current_length += len(word) + 1 
            current_chunk.append(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
    chunks.append(' '.join(current_chunk))
    return chunks

def create_zip_of_files(files):
    zip_name = "all_files.zip"
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for file in files:
            zipf.write(file)
    return zip_name

def get_zip_download_link(zip_file):
    with open(zip_file, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/zip;base64,{b64}" download="{zip_file}">Download All</a>'
    return href

def main():
    st.title(" DrLlama7B")
    prompt = f"Write ten funny jokes that are tweet length stories that make you laugh.  Show as markdown outline with emojis for each."
    example_input = st.text_input("Enter your example text:", value=prompt)
    if st.button("Run Prompt With Dr Llama"):
        try:
            StreamLLMChatResponse(example_input)
        except:
            st.write('Dr. Llama is asleep.  Starting up now on A10 - please give 5 minutes then retry as KEDA scales up from zero to activate running container(s).')
    openai.api_key = os.getenv('OPENAI_KEY')
    menu = ["txt", "htm", "xlsx", "csv", "md", "py"]
    choice = st.sidebar.selectbox("Output File Type:", menu)
    model_choice = st.sidebar.radio("Select Model:", ('gpt-3.5-turbo', 'gpt-3.5-turbo-0301'))
    filename = save_and_play_audio(audio_recorder)
    if filename is not None:
        transcription = transcribe_audio(key, filename, "whisper-1")
        st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)
        filename = None
    user_prompt = st.text_area("Enter prompts, instructions & questions:", '', height=100)
    collength, colupload = st.columns([2,3])  # adjust the ratio as needed
    with collength:
        max_length = st.slider("File section length for large files", min_value=1000, max_value=128000, value=12000, step=1000)
    with colupload:
        uploaded_file = st.file_uploader("Add a file for context:", type=["pdf", "xml", "json", "xlsx", "csv", "html", "htm", "md", "txt"])
    document_sections = deque()
    document_responses = {}
    if uploaded_file is not None:
        file_content = read_file_content(uploaded_file, max_length)
        document_sections.extend(divide_document(file_content, max_length))
    if len(document_sections) > 0:
        if st.button("👁️ View Upload"):
            st.markdown("**Sections of the uploaded file:**")
            for i, section in enumerate(list(document_sections)):
                st.markdown(f"**Section {i+1}**\n{section}")
        st.markdown("**Chat with the model:**")
        for i, section in enumerate(list(document_sections)):
            if i in document_responses:
                st.markdown(f"**Section {i+1}**\n{document_responses[i]}")
            else:
                if st.button(f"Chat about Section {i+1}"):
                    st.write('Reasoning with your inputs...')
                    response = chat_with_model(user_prompt, section, model_choice)
                    st.write('Response:')
                    st.write(response)
                    document_responses[i] = response
                    filename = generate_filename(f"{user_prompt}_section_{i+1}", choice)
                    create_file(filename, user_prompt, response, should_save)
                    st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)
    if st.button('💬 Chat'):
        st.write('Reasoning with your inputs...')
        user_prompt_sections = divide_prompt(user_prompt, max_length)
        full_response = ''
        for prompt_section in user_prompt_sections:
            response = chat_with_model(prompt_section, ''.join(list(document_sections)), model_choice)
            full_response += response + '\n'  # Combine the responses
        response = full_response
        st.write('Response:')
        st.write(response)
        filename = generate_filename(user_prompt, choice)
        create_file(filename, user_prompt, response, should_save)
        st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)
    all_files = glob.glob("*.*")
    all_files = [file for file in all_files if len(os.path.splitext(file)[0]) >= 20]  # exclude files with short names
    all_files.sort(key=lambda x: (os.path.splitext(x)[1], x), reverse=True)  # sort by file type and file name in descending order
    if st.sidebar.button("🗑 Delete All"):
        for file in all_files:
            os.remove(file)
        st.experimental_rerun()
    if st.sidebar.button("⬇️ Download All"):
        zip_file = create_zip_of_files(all_files)
        st.sidebar.markdown(get_zip_download_link(zip_file), unsafe_allow_html=True)
    file_contents=''
    next_action=''
    for file in all_files:
        col1, col2, col3, col4, col5 = st.sidebar.columns([1,6,1,1,1])  # adjust the ratio as needed
        with col1:
            if st.button("🌐", key="md_"+file):  # md emoji button
                with open(file, 'r') as f:
                    file_contents = f.read()
                    next_action='md'
        with col2:
            st.markdown(get_table_download_link(file), unsafe_allow_html=True)
        with col3:
            if st.button("📂", key="open_"+file):  # open emoji button
                with open(file, 'r') as f:
                    file_contents = f.read()
                    next_action='open'
        with col4:
            if st.button("🔍", key="read_"+file):  # search emoji button
                with open(file, 'r') as f:
                    file_contents = f.read()
                    next_action='search'
        with col5:
            if st.button("🗑", key="delete_"+file):
                os.remove(file)
                st.experimental_rerun()
    if len(file_contents) > 0:
        if next_action=='open':
            file_content_area = st.text_area("File Contents:", file_contents, height=500)
        if next_action=='md':
            st.markdown(file_contents)
        if next_action=='search':
            file_content_area = st.text_area("File Contents:", file_contents, height=500)
            st.write('Reasoning with your inputs...')
            response = chat_with_model(user_prompt, file_contents, model_choice)
            filename = generate_filename(file_contents, choice)
            create_file(filename, user_prompt, response, should_save)
            st.experimental_rerun()


    # Feedback
    # Step: Give User a Way to Upvote or Downvote
    feedback = st.radio("Step 8: Give your feedback", ("👍 Upvote", "👎 Downvote"))

    if feedback == "👍 Upvote":
        st.write("You upvoted 👍. Thank you for your feedback!")
    else:
        st.write("You downvoted 👎. Thank you for your feedback!")

load_dotenv()
st.write(css, unsafe_allow_html=True)
st.header("Chat with documents :books:")
user_question = st.text_input("Ask a question about your documents:")
if user_question:
    process_user_input(user_question)
with st.sidebar:
    st.subheader("Your documents")
    docs = st.file_uploader("import documents", accept_multiple_files=True)
    with st.spinner("Processing"):
        raw = pdf2txt(docs)
        if len(raw) > 0:
            length = str(len(raw))
            text_chunks = txt2chunks(raw)
            vectorstore = vector_store(text_chunks)
            st.session_state.conversation = get_chain(vectorstore)
            st.markdown('# AI Search Index of Length:' + length + ' Created.')  # add timing
            filename = generate_filename(raw, 'txt')
            create_file(filename, raw, '', should_save)

if __name__ == "__main__":
    main()

```
