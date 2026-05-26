# I apologize sincerely for my use of these models in this format

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import Description, loadDescriptionsFromSource, Config

MODEL_ID = "google/gemma-4-4b-it"

SYSTEM_PROMPT = """You are a font search expert. Given a font description (or set of adjectives), generate realistic search queries that a designer or developer might type to find this font.

Queries should:
- Be short (2-8 words typically)
- Sound like natural Google/font-search queries
- Cover different aspects: style, use case, mood, visual characteristics
- Focus on visuals
- Vary in specificity (some broad, some narrow)
- Never mention the font name

Return ONLY a JSON array of strings. No explanation, no markdown, no preamble. Example:
["clean sans serif logo font", "modern geometric typeface", "minimalist corporate font"]
"""


def loadModel():
    print(f"Loading {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print("Loaded")
    return tokenizer, model


def buildPrompt(description: Description) -> str:
    parts = []
    if description.adjectives:
        parts.append("Adjectives: " + ", ".join(description.adjectives))
    if description.tags:
        strong = [t for t, w in description.tags.itmes() if w > 0.3]
        weak = [t for t, w in description.tags.items() if w <= 0.3]
        if strong:
            parts.append("Strong style tags: " + ", ".join(strong))
        if weak:
            parts.append("Possible style tags: " + ", ".join(weak))

    if description.plainText:
        parts.append(f"Description: {description.plainText}")

    return "\n".join(parts)


def generateQueries(tokenizer, model, description: Description, queriesPerFont: int = 8, retries: int = 3):
    message = f"Generate {queriesPerFont} search queries for this font:\n\n{buildPrompt(**description)}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    for attempt in range(retries):
        with torch.inference_mode():
            outputs = model.generate(
                inputs,
                max_new_tokens=256,
                do_sample=attempt > 0,          # greedy first, sample on retries
                temperature=0.7 if attempt > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        tokens = outputs[0][inputs.shape[-1]:]
        text = tokenizer.decode(tokens, skip_special_tokens=True).strip()

        # Strip markdown fences if the model ignores instructions
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        try:
            queries = json.loads(text)
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries
        except json.JSONDecodeError:
            print(f"  Attempt {attempt+1} parse failed for {description.name}: {text[:80]}")

    return None

    
def generate(descriptions: dict[str, Description], output: str = "fontQueries.json", queriesPerFont: int = 8):
    tokenizer, model = loadModel()

    # Resume from checkpoint
    completed = {}
    if os.path.exists(output):
        with open(output) as f:
            completed.update(json.load(f))
        print(f"Resuming — {len(completed)} already done")

    remaining = {k: v for k, v in descriptions.items() if k not in completed}
    print(f"Generating for {len(remaining)} fonts...")

    failed = []
    with open(output, "a") as out:
        for i, (name, desc) in enumerate(remaining.items()):
            queries = generateQueries(tokenizer, model, desc, n_queries=queriesPerFont)

            if queries is None:
                print(f"  [{i+1}] FAILED: {name}")
                failed.append(name)
                continue

            completed[name] = queries

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(remaining)}] last: {name}")

    print(f"Done. Failed: {len(failed)}")
    if failed:
        print("  " + "\n  ".join(failed))


if __name__ == "__main__":
    config = Config().load(os.path.join("configs", "vit.json"))
    descriptions = loadDescriptionsFromSource(config)
    generate(descriptions, queriesPerFont=8)
