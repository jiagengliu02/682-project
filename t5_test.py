from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the tokenizer
# tokenizer = T5Tokenizer.from_pretrained('deep-learning-analytics/GrammarCorrector', legacy=False)
tokenizer = T5Tokenizer.from_pretrained('deep-learning-analytics/GrammarCorrector')
print(tokenizer.vocab_size)

# Load the model
model = T5ForConditionalGeneration.from_pretrained('deep-learning-analytics/GrammarCorrector')

def correct_grammar(input_text):
    # Tokenize the input text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=64, truncation=True)

    # Generate corrected text
    outputs = model.generate(inputs, max_length=64, num_return_sequences=1)

    # Decode the generated text
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

# Example usage

sentences = [
    "She don't like to eat vegetables.",
    "He go to the store every day.",
    "They was playing soccer in the park.",
    "I can speaks three languages.",
    "The cat chased it's tail.",
    "We was late to the meeting.",
    "She have a beautiful voice.",
    "He don't know how to swim.",
    "The book are on the table.",
    "I seen him at the party last night.",
]

for input_text in sentences:
    print(input_text)
    corrected_text = correct_grammar(input_text)
    print(corrected_text)
