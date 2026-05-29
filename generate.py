# I apologize sincerely for my use of these models in this format. This is wasteful and grotesque

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import Description, loadGoogleDescriptions, loadDescriptionsFromSource, Config

MODEL_ID = "microsoft/Phi-4"

# SYSTEM_PROMPT = """You are a font search expert. Given a font description (or set of adjectives), generate realistic search queries that a designer or developer might type to find this font.

# Queries should:
# - Be short (2-8 words typically)
# - Sound like natural Google/font-search queries
# - Cover different aspects: style, use case, mood, visual characteristics
# - Focus on visuals
# - Vary in specificity and length (some broad, some narrow)
# - Never mention the font name

# Return ONLY a JSON array of strings. No explanation, no markdown, no preamble. Example:
# ["clean sans serif logo font", "modern geometric typeface with monolinear strokes", "minimalist corporate font balancing technical precision with approachable rounded details"]
# """


def loadModel():
    print(f"Loading {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
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
        strong = [t for t, w in description.tags.items() if w > 0.3]
        weak = [t for t, w in description.tags.items() if w <= 0.3]
        if strong:
            parts.append("Strong style tags: " + ", ".join(strong))
        if weak:
            parts.append("Possible style tags: " + ", ".join(weak))

    if description.plainText:
        parts.append(f"Description: {description.plainText}")

    return "\n".join(parts)

import json
import numpy as np

def buildContrastivePrompt(
    description: Description,
    allDescriptions: dict[str, Description],
    n_similar: int = 4,
) -> str:
    # Find similar fonts by tag overlap to use as contrast examples
    def tagOverlap(a: Description, b: Description) -> float:
        if not a.tags or not b.tags:
            return 0.0
        keys = set(a.tags) | set(b.tags)
        overlap = sum(min(a.tags.get(k, 0), b.tags.get(k, 0)) for k in keys)
        union   = sum(max(a.tags.get(k, 0), b.tags.get(k, 0)) for k in keys)
        return overlap / union if union else 0.0

    similar = sorted(
        [(name, desc) for name, desc in allDescriptions.items() if name != description.name],
        key=lambda x: tagOverlap(description, x[1]),
        reverse=True
    )[:n_similar]

    parts = []
    if description.adjectives:
        parts.append("Adjectives: " + ", ".join(description.adjectives))
    if description.tags:
        strong = [t for t, w in description.tags.items() if w > 0.3]
        weak   = [t for t, w in description.tags.items() if w <= 0.3]
        if strong:
            parts.append("Strong style tags: " + ", ".join(strong))
        if weak:
            parts.append("Possible style tags: " + ", ".join(weak))
    if description.plainText:
        parts.append(f"Description: {description.plainText}")

    if similar:
        parts.append("\nFonts in a similar style category (your queries must distinguish from these):")
        for name, sim in similar:
            simAdj = ", ".join(sim.adjectives[:6]) if sim.adjectives else ""
            simTags = ", ".join(t for t, w in sim.tags.items() if w > 0.3) if sim.tags else ""
            parts.append(f"  - {name}: {simAdj} | tags: {simTags}")

    return "\n".join(parts)


SYSTEM_PROMPT = """You are a font search expert. Given a font and a list of visually similar fonts, generate realistic search queries a designer might type to find exactly this font and not the similar ones listed.

Queries must:
- Be long and descriptive enough to be specific (aim for 8-20 words for at least half of them)
- Describe visual qualities that make this font unique within its style category — specific details like stroke contrast, terminal style, weight distribution, x-height, historical period, construction geometry, or emotional register
- Include some broader category queries (3-6 words) for general discoverability
- Sound like natural search queries, not academic descriptions
- Never mention the font name
- Do not make assumptions about data you do not have access to

Think carefully about what makes this font visually distinct from the similar fonts listed. A query that would equally describe one of those similar fonts is a bad query.

Return ONLY a JSON array of strings. No explanation, no markdown, no preamble. Example:
["art deco sans serif font", "counterless geometric art deco typeface with uniform stroke weight", "compressed 1930s display font where the A has no crossbar and the O is perfectly circular", "stark geometric headline font with closed counters and minimal stroke contrast suitable for luxury packaging"]"""


def parseOutput(text: str) -> list[str] | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        queries = json.loads(text)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            return queries
    except json.JSONDecodeError:
        pass
    return None


def generateBatch(
    tokenizer,
    model,
    batch: list[tuple[str, Description]],
    allDescriptions: dict[str, Description],
    queriesPerFont: int = 8,
) -> dict[str, list[str] | None]:
    prompts = []
    for _, desc in batch:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": 
                f"Generate {queriesPerFont} search queries for this font:\n\n"
                f"{buildContrastivePrompt(desc, allDescriptions)}"
            },
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        ))

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=384 * 2,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = {}
    prompt_len = inputs["input_ids"].shape[-1]
    for i, (name, _) in enumerate(batch):
        new_tokens = outputs[i][prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results[name] = parseOutput(text)

    return results


def generate(
    descriptions: dict[str, Description],
    output: str = os.path.join("results", "fontQueries.json"),
    queriesPerFont: int = 8,
    batchSize: int = 8,
    maxRetries: int = 2,
):
    tokenizer, model = loadModel()

    completed = {}
    if os.path.exists(output):
        with open(output) as f:
            completed.update(json.load(f))
        print(f"Resuming — {len(completed)} already done")

    remaining = [(k, v) for k, v in descriptions.items() if k not in completed]
    print(f"Generating for {len(remaining)} fonts...")

    try:
        for attempt in range(maxRetries + 1):
            if not remaining:
                break
            if attempt > 0:
                print(f"Retry pass {attempt} for {len(remaining)} failed fonts...")

            still_failing = []
            for batchStart in range(0, len(remaining), batchSize):
                batch = remaining[batchStart : batchStart + batchSize]
                results = generateBatch(tokenizer, model, batch, descriptions, queriesPerFont)

                for name, queries in results.items():
                    if queries is None:
                        still_failing.append((name, descriptions[name]))
                    else:
                        completed[name] = queries

                print(f"\r  [{min(batchStart + batchSize, len(remaining))}/{len(remaining)}]", end="")

            print()
            remaining = still_failing

    except KeyboardInterrupt:
        pass
    finally:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w") as f:
            json.dump(completed, f, indent=2)
        print(f"Saved {len(completed)} entries. Still failing: {len(remaining)}")


if __name__ == "__main__":
    config = Config().load(os.path.join("configs", "vit.json"))
    descriptions = loadDescriptionsFromSource(config.dataset)
    generate(descriptions, queriesPerFont=8)
