from pathlib import Path

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "flax-community/papuGaPT2"
MODEL_DIR = Path("./models/checkpoint-7000-best-to-10000")

CONTEXT_SIZE = 200
SPEAKER_1_TAG = "<speaker1>"
SPEAKER_2_TAG = "<speaker2>"


def get_model(pytorch_model_path: Path):
    model = AutoModelForCausalLM.from_pretrained(pytorch_model_path)
    return model


def get_tokenizer(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # add special tokens - speaker1 and speaker2 tags to tokenizer
    SPEAKER_TOKENS = {"additional_special_tokens": [SPEAKER_1_TAG, SPEAKER_2_TAG]}
    tokenizer.add_special_tokens(SPEAKER_TOKENS)
    # set pad token as eos
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def make_pred(message: str, tokenizer, model):
    model_input = f"{message}{SPEAKER_1_TAG}"
    input_ids = tokenizer.encode(model_input, return_tensors="pt")
    outputs = model.generate(
        input_ids, do_sample=True, max_length=CONTEXT_SIZE + 100, top_k=50, top_p=0.95
    )
    response = tokenizer.decode(outputs[0])
    response = response.replace(message, "", 1)
    next_speaker_generated_idx = response.find(SPEAKER_2_TAG)
    if next_speaker_generated_idx >= 0:
        response = response[:next_speaker_generated_idx]
    return response


def get_message_with_context(history):
    return "".join(history)[-CONTEXT_SIZE:]


def main():
    st.set_page_config(layout="wide")
    # preserve data in session_state
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "model" not in st.session_state:
        st.session_state["model"] = get_model(MODEL_DIR)
    if "tokenizer" not in st.session_state:
        st.session_state["tokenizer"] = get_tokenizer()

    message = st.text_input(f"Type a message (you are {SPEAKER_2_TAG}):")
    if message:
        message_with_speaker = f"{SPEAKER_2_TAG} {message}"
        st.session_state["history"].append(message_with_speaker)

        message_with_context = get_message_with_context(st.session_state["history"])
        with st.spinner("Typing the answer..."):
            model_pred = make_pred(
                message_with_context,
                st.session_state["tokenizer"],
                st.session_state["model"],
            )
        st.session_state["history"].append(model_pred)

    st.text("\n".join(st.session_state.history))

    if st.button("reset history"):
        st.session_state["history"] = []


if __name__ == "__main__":
    main()
