# YouTube Content Generator with Hugging Face's Gemma

## Overview
This project is a **YouTube Content Generator** that uses Hugging Face's **Gemma** model to generate engaging YouTube video titles and scripts. The application is built using **Streamlit** for a user-friendly interface and leverages **LangChain** for conversational memory and prompt management.

## Features
1. **Content Generator**
   - Input a topic and generate:
     - A concise YouTube video title.
     - A detailed YouTube video script.
   - Fetch relevant Wikipedia research to enrich the script.
   
2. **Test API Mode**
   - Test Hugging Face's Inference API with custom queries to understand the model's response capabilities.

3. **Temperature Control**
   - Adjustable temperature slider to control creativity levels in model outputs:
     - Low values (e.g., 0.1) for deterministic outputs.
     - High values (e.g., 0.9) for creative outputs.

4. **Streamlit Interface**
   - Interactive, responsive UI with sidebar navigation and expandable sections for history and research.

## Project Structure
```
project/
├── wiki_researcher.py                  # Main Streamlit application file.
├── utils/
│   ├── api.py             # Functions for interacting with the Hugging Face API.
│   ├── memory.py          # Memory management using LangChain.
│   ├── templates.py       # Prompt templates for generating content.
│   ├── constants.py       # API URL and headers.
│   └── wiki.py            # Wrapper for Wikipedia utilities.
├── requirements.txt        # Python dependencies.
└── README.md               # Project documentation.
```

## Setup

### Prerequisites
- Python 3.8 or higher.
- A Hugging Face account with an API token.
- Streamlit installed.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/UsamaYousaf/hugging_scripts.git
   cd hugging_scripts
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.streamlit/secrets.toml` file to store your Hugging Face API token:
   ```toml
   [secrets]
   auth_token = "your_huggingface_api_token"
   ```

### Running the Application
1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open the provided URL in your browser to access the application.

## Usage
- **Content Generator**: Enter a topic in the input field to generate titles and scripts. Adjust the temperature slider for creativity control.
- **Test API**: Use the "Test API" mode to send custom queries to the Hugging Face API.

## Technologies Used
- **Streamlit**: For building the interactive web application.
- **LangChain**: For managing memory and prompt templates.
- **Hugging Face API**: To interact with the Gemma model.

## Contribution
Contributions are welcome! If you have ideas or suggestions, feel free to open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

---
For more details, check the [GitHub Repository](https://github.com/UsamaYousaf/hugging_scripts).

