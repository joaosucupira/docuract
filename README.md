[![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-brightgreen?logo=streamlit)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gemini API](https://img.shields.io/badge/Gemini%20API-Required-orange)](https://ai.google.dev/)

# Docuract Hub

Docuract Hub is a Streamlit-based web application that allows you to interact with the content of your PDF documents using conversational AI powered by Gemini (Google Generative AI) and HuggingFace embeddings. Upload your PDFs, process them, and ask questions about their content in natural language.

---

## Features

- Upload multiple PDF documents.
- Extract and split text from PDFs.
- Conversational interface to query document content.
- Uses Gemini (Google Generative AI) for responses.
- Embedding via HuggingFace's Instructor model.

---

## How to Run

1. **Clone the repository** and navigate to the project folder.

2. **Install dependencies**:

   ```
   pip install -r requirements.txt
   ```

3. **Set up your environment variables**:
- Create a `.env` file in the root directory.
- Add your Google Gemini API key:
  ```
  GOOGLE_API_KEY=your_google_gemini_api_key
  ```

4. **Start the app**:

   ```
   streamlit run app.py
   ```

3. **Set up your environment variables**:
- Create a `.env` file in the root directory.
- Add your Google Gemini API key:
  ```
  GOOGLE_API_KEY=your_google_gemini_api_key
  ```

4. **Start the app**:



5. **Usage**:
- Upload your PDF documents using the sidebar.
- Click "Process" to extract and embed the text.
- Ask questions about your documents in the main chat interface.

---

## Screenshot

<a href="https://ibb.co/GQVLVwTS"><img src="https://i.ibb.co/Kx9k9Zwv/docuract-img.jpg" alt="docuract-img" border="0"></a>)

---

