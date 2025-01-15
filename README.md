
# AI Mental Health Chatbot

This tool provides a mental health chatbot that helps users with their emotional well-being. It uses a combination of AI-powered language models and vector databases to provide thoughtful and context-aware responses. Users can ask questions about mental health, and the chatbot will use its knowledge base to provide relevant, empathetic advice and support.

## Technical Architecture

The technical architecture of the AI Mental Health Chatbot follows a structured flow: The User Interface collects the user's input, which is passed to the LLM (ChatGroq) to generate a response. The LLM queries the Vector Database (Chroma DB) for relevant context and documents. These documents are then processed by the RetrievalQA Chain, which combines the retrieved context with the LLM's capabilities to generate an accurate and empathetic response. Finally, the response is displayed back to the user in the Response Output section of the UI.
```bash
       +-------------------+
       |   User Interface  |   <-- User Input (text)
       +-------------------+
               |
               v
       +-------------------+
       |   LLM (ChatGroq)  |   <-- Pass User Query
       +-------------------+
               |
               v
       +-------------------+
       | Vector Database   |   <-- Search for Context
       |  (Chroma DB)      |   <-- Retrieves Documents
       +-------------------+
               |
               v
       +-------------------+
       | RetrievalQA Chain |   <-- Uses LLM and Documents to Generate Response
       +-------------------+
               |
               v
       +-------------------+
       |   Response Output |   <-- Display Answer in UI
       +-------------------+

```

# Set-up

1. To get started we first need to get an API_KEY from here: https://console.groq.com/keys. Inside `.env` update the value of `GROQ_API_KEY` with the API_KEY you created. 

2. Run the streamlit app:
   ```commandline
   streamlit run ./app/main.py

   ```