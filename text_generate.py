import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Streamlit app
def main():
    st.title('Text Generation with Streamlit')

    # Model selection
    model_choice = st.radio("Select a Model:", ('Poem Generation Model', 'Harry Potter Model'))

    # Load the selected tokenizer
    if model_choice == 'Poem Generation Model':
        tokenizer_path = 'tokenizer.pkl'
    else:  # Assuming 'Harry Potter Model' is the other choice
        tokenizer_path = 'tokenizer.pk2'
    
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Load the selected text generation model
    if model_choice == 'Text Generation Model':
        model_path = 'poem_generation_model.h5'
    else:  # Assuming 'Harry Potter Model' is the other choice
        model_path = 'harrypotter_model.h5'
    
    model = load_model(model_path)
    MAX_SEQUENCE_LENGTH = 40

    # Text generation function
    def generate_text(seed_text, max_words=200):
        generated_text = seed_text
        for _ in range(max_words):
            input_sequence = tokenizer.texts_to_sequences([seed_text])
            input_sequence = pad_sequences(input_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')

            predicted_probabilities = model.predict(input_sequence)
            predicted_index = np.argmax(predicted_probabilities, axis=-1)
            predicted_word = tokenizer.index_word[predicted_index[0]]

            generated_text += ' ' + predicted_word
            seed_text = seed_text + ' ' + predicted_word

        return generated_text

    # User input and text generation controls
    seed_text = st.text_input('Enter your seed text:')
    num_words = st.number_input('Number of words to generate:', min_value=10, max_value=200, value=50, step=10)

    if st.button('Generate Text'):
        generated_text = generate_text(seed_text, num_words)
        st.write('Generated Text:')
        st.write(generated_text)


if __name__ == '__main__':
    main()
