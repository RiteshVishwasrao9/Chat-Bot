import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load pre-trained sequence-to-sequence model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

def generate_response(input_text):
    # Use the model to generate a response
    inputs = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors='pt'
    )
    outputs = model.generate(inputs['input_ids'], num_beams=4, no_repeat_ngram_size=2)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Create a Streamlit app
st.title("AI Assistant")

# Get user input
user_input = st.text_area("Type your question or prompt", height=200)

if st.button("Ask AI"):
    # Generate a response using the model
    response = generate_response(user_input)
    st.write(response)

# Add some basic styling to make it look nicer
st.markdown("<style>body {background-color: #f0f0f0;}</style>", unsafe_allow_html=True)

# Add a button to open YouTube video (example of integrating external functionality)
if st.button("Open YouTube"):
    import webbrowser
    webbrowser.open("https://www.youtube.com")