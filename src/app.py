import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# App title
st.title("Socratic Chat")

# Sidebar for settings
with st.sidebar:
    st.header("Model Settings")
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.1)
    max_length = st.slider("Max Length", 64, 2048, 512, 64)

# Load model
@st.cache_resource
def load_model():
    base_model = "meta-llama/Meta-Llama-3-8B"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(model, "models/socratic-llama-3-8b")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    return model, tokenizer

model, tokenizer = load_model()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Type your message..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            assistant_response = response.split("assistant")[-1].strip()
            
            st.markdown(assistant_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})