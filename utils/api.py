def query_huggingface(prompt, temperature):
    API_URL = "https://api-inference.huggingface.co/models/google/gemma-7b"
    HEADERS = {
        "authorization": f"Bearer {st.secrets['auth_token']}",
        "content-type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 200,
            "temperature": temperature,
            "top_k": 50,
            "top_p": 0.9
        }
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    result = response.json()
    return result[0]["generated_text"] if result else "Error generating response."