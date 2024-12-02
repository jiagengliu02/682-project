from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id = tokenizer.eos_token_id

# Load the model
model = GPT2LMHeadModel.from_pretrained('gpt2')
input_text = "Please correct the grammar errors in this sentence: "
print(model.transformer.embed_dim)

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
# input_text = "How old are you?"
for i in range(len(sentences)):
    input_ids = tokenizer.encode(input_text + sentences[i], return_tensors='pt')
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    print(attention_mask)

    # Generate text
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=50,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # stop_sequence=["\n"],
        # repetition_penalty=2.0,
    )

    # Decode the generated text
    print(output[0])
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)
