
import streamlit as st
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.llms import OpenAI
import os


# Set the page title and other configurations
st.set_page_config(
    page_title="Chat & Heal",  # Custom title
    page_icon="🧠",  # Custom icon (emoji or image)
    layout="wide"  # Optional: "wide" for full-screen layout, "centered" for default
)
try:
    from langchain_groq import ChatGroq
except ImportError:
    raise ImportError("ChatGroq is not a recognized class in LangChain. Ensure it's correctly implemented or imported.")


def initialize_llm():
    import os
    # Directly fetch the GROQ API key from environment variables
    # api_key = os.getenv("GROQ_API_KEY")

    try:
        # Initialize the ChatGroq instance
        llm = ChatGroq(
            temperature=0,
            groq_api_key='API-KEY',
            model_name="llama-3.3-70b-versatile"
        )
    except Exception as e:
        st.error(f"Failed to initialize ChatGroq: {e}")
        st.stop()

    return llm

# Function to create a vector database
def create_vector_db():
    try:
        loader = DirectoryLoader("app/resource/", glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text = text_splitter.split_documents(documents)
        embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_db = Chroma.from_documents(text, embeddings, persist_directory='./chroma_db')
        vector_db.persist()
        st.success("Vector Database created and data saved!")
        return vector_db
    except Exception as e:
        st.error(f"Error creating vector database: {e}")
        st.stop()


# Function to load the vector database
def load_vector_db(db_path):
    try:
        embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        st.sidebar.success("Vector Database Loaded.")
        return vector_db
    except Exception as e:
        st.error(f"Error loading vector database: {e}")
        st.stop()


# Function to set up the QA chain
def setup_qa_chain(vector_db, llm):
    prompt_template = """You are a compassionate mental health chatbot. Respond thoughtfully to the following question:
    {context}
    User: {question}
    Chatbot: """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain


# Function to update the chat history
def chatbot_response(user_input, qa_chain, history):
    if not user_input.strip():
        return "Please provide a valid input", history

    # Run the QA chain to get the response
    response = qa_chain.run(user_input)

    # Add the user input and response to the history
    history.append(("User: " + user_input, "Your Buddy: " + response))

    return response, history


# Streamlit main function
def main():
    st.title("🤖 Chat & Heal: Your Mental Health Guide🧘‍♀️")
    st.write("This chatbot is designed to provide thoughtful responses to your mental health-related questions.")

    # Initialize the LLM
    llm = initialize_llm()

    # Vector database setup
    db_path = "./chroma_db"
    if not os.path.exists(db_path):
        st.sidebar.write("Creating Vector Database...")
        vector_db = create_vector_db()
    else:
        vector_db = load_vector_db(db_path)

    # Set up QA chain
    qa_chain = setup_qa_chain(vector_db, llm)

    # Initialize session state for storing chat history if not already done
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Display chat history
    if st.session_state.history:
        st.subheader("💬 Chat History")
        for user_msg, bot_msg in st.session_state.history:
            st.write(user_msg)
            st.write(bot_msg)

    # Chat UI
    st.subheader("💬 Chat with Your AI friend")
    user_input = st.text_input("Ask your question (type 'exit' to quit):", "")

    if user_input.lower() == "exit":
        st.write("Chatbot: Take care of yourself, goodbye!")
    elif user_input:
        try:
            with st.spinner("Thinking..."):
                # Get response and update history
                response, history = chatbot_response(user_input, qa_chain, st.session_state.history)

                # Update session state with the new chat history
                st.session_state.history = history

                # Display only the latest response
                st.write(f"User: {user_input}")
                st.write(f"Chatbot: {response}")  # Show the latest response

        except Exception as e:
            st.error(f"Error generating response: {e}")


# Run the app
if __name__ == "__main__":
    main()
