import streamlit as st
import requests
import uuid

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())

st.title("RAG Chatbot")

# Display chat messages from history on app rerun
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.write(message['content'])

# Accept user input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to history
    st.session_state['messages'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Prepare payload
    payload = {
        "question": prompt,
        "session_id": st.session_state['session_id']
    }

    # Send request to backend
    try:
        response = requests.post("http://localhost:8000/generate", json=payload)
        response.raise_for_status()
        answer = response.json().get('answer', '')

        # Add assistant message to history
        st.session_state['messages'].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
