import gradio as gr
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import requests
from gtts import gTTS
import speech_recognition as sr
import os

# Assurez-vous d'avoir installé toutes les bibliothèques nécessaires
# pip install gradio langchain chromadb ocrmypdf pypdf requests gtts speech_recognition

# Étape 1 : Charger le contenu du PDF
def load_pdf_content(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Erreur lors du chargement du PDF: {e}")
        return []

# Étape 2 : Créer un pipeline de question-réponse avec LangChain et GPT-4
def create_qa_pipeline(documents, api_key):
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectordb = Chroma.from_documents(documents, embeddings)
        prompt_template = PromptTemplate(
            input_variables=["question", "context"],
            template="""
            Question: {question}
            Context: {context}
            Answer: """,
        )
        llm = OpenAI(model_name="text-davinci-003", api_key=api_key)
        chain = LLMChain(prompt_template=prompt_template, llm=llm)
        return chain, vectordb
    except Exception as e:
        print(f"Erreur lors de la création du pipeline QA: {e}")
        return None, None

# Étape 3 : Traiter la question et retourner la réponse
def answer_question(question, pdf_path, api_key):
    documents = load_pdf_content(pdf_path)
    if not documents:
        return "Erreur lors du chargement du document PDF."
    
    qa_chain, vectordb = create_qa_pipeline(documents, api_key)
    if not qa_chain or not vectordb:
        return "Erreur lors de la configuration du pipeline QA."

    try:
        context = vectordb.similarity_search(question)
        if not context:
            return "Aucun contexte pertinent trouvé dans le PDF."
        result = qa_chain.run(question=question, context=context[0].page_content)
        return result
    except Exception as e:
        print(f"Erreur lors de la recherche de similarité ou de l'exécution du pipeline: {e}")
        return "Erreur lors de la recherche de la réponse."

# Fonction pour convertir la réponse en audio
def convert_text_to_audio(text):
    try:
        tts = gTTS(text=text, lang='ar')
        audio_file = "response.mp3"
        tts.save(audio_file)
        return audio_file
    except Exception as e:
        print(f"Erreur lors de la conversion texte en audio: {e}")
        return None

# Fonction principale de traitement des fichiers
def process_files(audio_file, pdf_file):
    # Reconnaissance vocale pour convertir l'audio en texte
    recognizer = sr.Recognizer()
    audio_data = sr.AudioFile(audio_file)
    with audio_data as source:
        audio_text = recognizer.record(source)

    try:
        question_text = recognizer.recognize_google(audio_text, language='ar-SA')
    except sr.UnknownValueError:
        return "Impossible de reconnaître l'audio", None
    except sr.RequestError as e:
        return f"Erreur lors de la reconnaissance vocale : {e}", None

    # Obtenir la clé d'API à partir de l'environnement ou directement
    api_key = "sk-proj-EUL0LnBMfuIfs1Xh1tiJT3BlbkFJ54ggV06FEnqeYPfv0hgQ"
    if not api_key:
        return "Clé API OpenAI non trouvée. Veuillez définir la clé API dans les variables d'environnement ou passer directement.", None

    # Répondre à la question en utilisant LangChain et GPT-4
    answer_text = answer_question(question_text, pdf_file.name, api_key)

    # Convertir la réponse en audio
    audio_path = convert_text_to_audio(answer_text)
    
    return answer_text, audio_path

# Créer l'interface Gradio
iface = gr.Interface(
    fn=process_files,
    inputs=[
        gr.Audio(type="filepath", label="Enregistrez votre question"),
        gr.File(label="Téléchargez le fichier PDF")
    ],
    outputs=[
        gr.Textbox(label="Réponse"),
        gr.Audio(label="Réponse Audio", type="filepath")
    ],
    title="Question Answering à partir d'un fichier PDF et d'un audio",
    description="Téléchargez un fichier PDF et enregistrez une question audio. L'application convertira l'audio en texte, trouvera la réponse dans le fichier PDF et la lira à haute voix."
)

# Lancer l'application
if __name__ == "__main__":
    iface.launch()
