import streamlit as st
import os
import json
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

a=6
st.set_page_config(page_icon="💰", page_title="BankingBuddy.ai", layout="centered")

# Inject custom CSS for animations and styling
st.markdown("""
<style>
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}
h1, p, .stButton, .stMarkdown {
  animation: fadeIn 0.6s ease-in-out;
}
hr {
  border: 1px solid #1E3A8A;
}
.chatbox {
  background-color: #F8FAFC;
  padding: 1rem;
  border-radius: 0.5rem;
  box-shadow: 0 1px 5px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    """
    <h1 style='text-align: center; color: #1E3A8A;'>BankingBuddy.ai : Your Smart Financial Assistant</h1>
    <p style='text-align: center; font-size:18px;'>Ask about loans, savings, accounts or banking services — get instant answers!</p>
    <hr/>
    """, 
    unsafe_allow_html=True
)

# Sidebar actions
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

with col2:
    if st.button("📋 View History"):
        st.session_state.show_history = not st.session_state.get('show_history', False)

# Initialize session state
if 'prompt_history' not in st.session_state:
    st.session_state.prompt_history = []

if 'show_history' not in st.session_state:
    st.session_state.show_history = False

# Display history if toggled
if st.session_state.show_history:
    st.sidebar.markdown("### 📋 Chat History")
    if st.session_state.prompt_history:
        if st.sidebar.button("🗑️ Clear History"):
            st.session_state.prompt_history = []
            st.rerun()
        for i, entry in enumerate(reversed(st.session_state.prompt_history[-10:])):
            with st.sidebar.expander(f"Query {len(st.session_state.prompt_history) - i}: {entry['timestamp']}"):
                st.write("**Question:**")
                st.write(entry['prompt'])
                if entry.get('response'):
                    st.write("**Answer:**")
                    st.write(entry['response'])
                else:
                    st.write("*Response not available*")
    else:
        st.sidebar.info("No chat history available.")

st.markdown("""
<hr>
<div style='text-align: center; color: gray; font-size: 14px;'>
⚠️ <strong>Disclaimer:</strong> BankingBuddy.ai is an AI assistant trained on banking data and documents. 
This tool provides general financial information only and should not replace advice from your bank or financial advisor.
</div>
""", unsafe_allow_html=True)

# Save and update prompt history
def save_prompt_to_history(prompt):
    history_entry = {
        'prompt': prompt,
        'response': None,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.prompt_history.append(history_entry)
    return len(st.session_state.prompt_history) - 1

def update_history_with_response(index, response):
    if 0 <= index < len(st.session_state.prompt_history):
        st.session_state.prompt_history[index]['response'] = response

# Optional: Export history
def export_history_to_json():
    if st.session_state.prompt_history:
        return json.dumps(st.session_state.prompt_history, indent=2)
    return "No history to export"

if st.session_state.show_history and st.session_state.prompt_history:
    if st.sidebar.button("📥 Export History"):
        history_json = export_history_to_json()
        st.sidebar.download_button(
            label="💾 Download History JSON",
            data=history_json,
            file_name=f"banking_bot_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

def load_llm(GROQ_API_KEY, model_name="llama3-70b-8192"):
    return ChatGroq(api_key=GROQ_API_KEY, model_name=model_name, temperature=0.9, max_tokens=512)

def main():
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg['role']).markdown(msg['content'])

    prompt = st.chat_input('Ask me anything about banking, loans, savings, or finance...')

    if prompt:
        idx = save_prompt_to_history(prompt)

        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

    # Detect if it's only a greeting
        greeting_keywords = ["hi","hi!", "Hi", "Hi!", "hello", "hello!","Hello", "Hello!","hey","hey!","Hey", "Hey!", "hii","hii!","Hii","Hii!"]
        if prompt.strip().lower() in greeting_keywords:
           greeting_response = "Hello! 👋 Welcome to BankBuddy.ai. How can I assist you with your banking needs today?"
           st.chat_message('assistant').markdown(greeting_response)
           st.session_state.messages.append({'role': 'assistant', 'content': greeting_response})
           update_history_with_response(idx, greeting_response)
           return  # Skip calling the LLM

 
        CUSTOM_PROMPT = """
You are BankBuddy.ai — a professional and intelligent banking assistant.

Behavior Rules:
1. If the user input contains a **banking-related query**:
   - Use only the given context — do not make up information.
   - If the answer is not found in the context, reply with:
     **"I'm sorry, I couldn’t find that information in the available context."**

2. If the user asks about you or your identity (e.g., "Who are you?", "Are you a bot?", "What is BankBuddy?") reply with:
    - "Who are you?" : I'm BankBuddy.ai, your intelligent assistant designed to help you with everything from loans and credit cards to accounts, KYC, and banking policies.
    - "Are you a bot?" : Yes, your intelligent assistant designed to help you with everything from loans and credit cards to accounts etc..
    - "What is BankBuddy" : BankBuddy.ai is a AI assistant designed to help with bank related queries.

Maintain a respectful and professional tone. Keep responses clear, concise, and formatted properly.

---

📄 Context:
{context}

👤 User:
{question}

💬 BankBuddy.ai:


        """
        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load vectorstore.")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(GROQ_API_KEY=GROQ_API_KEY),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]

            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})
            update_history_with_response(idx, result)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            update_history_with_response(idx, f"Error occurred: {str(e)}")

    if st.session_state.prompt_history:
        st.sidebar.markdown(f"**Total Queries:** {len(st.session_state.prompt_history)}")

if __name__ == "__main__":
    main()

