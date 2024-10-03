import streamlit as st
import requests

st.title("My First RAG Chatbot")

st.write("Enter your query below:")

with st.form(key='query_form'):
    user_query = st.text_area('Query:', height=150)
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    if user_query.strip() == "":
        st.warning("Please enter a query.")
    else:
        with st.spinner('Processing...'):
            # Send the query to FastAPI server
            url = "http://localhost:8000/generate"
            payload = {"question": user_query}
            try:
                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    answer = response.json()['answer']
                    st.write("### Answer:")
                    st.write(answer)
                else:
                    st.error(f"Error: {response.status_code} {response.text}")
            except Exception as e:
                st.error(f"An error occurred: {e}")