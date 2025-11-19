from transformers import pipeline, AutoTokenizer
import itertools
import time


model_name = "distilgpt2"
generator = pipeline("text-generation", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompts = [
    "The future of ChatGPT is",
    "The cat is very old",
    "The secret to a healthy relationship is",
    "She love roses",
    " He is a gradmaster in chess"
]

max_lengths = [20, 50, 100]
temperatures = [0.5, 1.0, 1.5]
top_ks = [10, 50, 100]

combinations = list(itertools.product(max_lengths, temperatures, top_ks))

for prompt in prompts:
    print("=" * 80)
    print(f"PROMPT: {prompt}")
    print("=" * 80)

    for max_len, temp, topk in combinations:

        # ----- Timing -----
        start_time = time.time()
        output = generator(
            prompt,
            max_length=max_len,
            temperature=temp,
            top_k=topk,
            do_sample=True
        )
        end_time = time.time()
        elapsed = end_time - start_time

        text = output[0]["generated_text"]

        # ----- Token Counting -----
        token_ids = tokenizer.encode(text)
        num_tokens = len(token_ids)

        print(f"\n--- max_length={max_len}, temperature={temp}, top_k={topk} ---")
        print(f"Time taken: {elapsed:.4f} seconds")
        print(f"Token count: {num_tokens}")
        print(text)
        print("-" * 80)

    print("\n\n")


