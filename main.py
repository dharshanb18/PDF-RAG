
import streamlit as st
from agents.document_agent import DocumentAgent
from agents.query_agent import QueryAgent
from agents.response_agent import ResponseAgent

# Initialize agents
document_agent = DocumentAgent()
query_agent = QueryAgent(document_agent)
response_agent = ResponseAgent()

# Chat memory
if 'history' not in st.session_state:
    st.session_state.history = []

# Streamlit UI
st.title("Multilingual PDF Chat with OpenAI and FAISS")
st.write("Upload your PDF and have a conversation based on its content.")

# PDF upload
pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf:
    with st.spinner("Extracting text from PDF..."):
        document_agent.load_pdf(pdf)
        st.success("Text extraction complete!")

        # Display PDF text (show only the first 1000 characters)
        st.text_area("Extracted Text:", document_agent.get_extracted_text()[:1000] + '...')

        # Chat Interface
        question = st.text_input("Ask a question based on the PDF:")
        if question:
            # Retrieve relevant chunks and generate response
            relevant_chunks = query_agent.get_relevant_chunks(question)
            answer = response_agent.get_response(relevant_chunks, question)

            # Display top 3 results
            st.write(f"Answer: {answer}")

            # Store chat history
            st.session_state.history.append(f"Q: {question}\nA: {answer}")

            # Display chat memory
            st.write("Chat History:")
            for dialogue in st.session_state.history:
                st.write(dialogue)
