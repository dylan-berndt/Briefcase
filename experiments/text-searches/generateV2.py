# Second-pass caption generator, regenerating descriptions for the WHOLE
# corpus from scratch (not just the uncaptioned remainder). generate.py
# (Phi-4, bf16, one font per conversation) is left untouched; this writes to
# a separate output file by default.
#
# Two changes from generate.py, per the trainTextMLP.py experiments showing
# held-out QUERY generalization (not font generalization) is the actual
# bottleneck -- i.e. the fix is more/better description diversity per font,
# not more fonts:
#
# 1. Model: Qwen3.5-4B (GGUF, 8-bit) served locally by the Unsloth Desktop
#    app, called here over its OpenAI-compatible /v1/chat/completions API.
#    Landed on this after several dead ends running the model ourselves --
#    see conversation history: no working C++ compiler on this machine
#    (rules out building llama.cpp from source), the only pre-quantized
#    checkpoint we found had an unfixable compressed-tensors version
#    conflict, and there's no official AWQ release for this size. Serving it
#    through Unsloth Desktop sidesteps all of that -- this script is just an
#    HTTP client. Requires the UNSLOTH_API_KEY environment variable (from
#    the app's Settings -> API) and the app running with the model loaded;
#    see connectUnsloth. The app's context window is 8192 tokens, which
#    caps how large GROUP_SIZE * TOKENS_PER_FONT (+ prompt) can get.
#
# 2. Prompting: instead of one font per conversation, GROUP_SIZE
#    *stylistically similar* fonts (by tag/adjective cosine similarity, see
#    groupFonts) are bundled into a single conversation, and the model is
#    told explicitly to vary sentence structure/length BETWEEN fonts in the
#    group, not just within one font's own queries. This keeps the same
#    contrastive framing generate.py used (describe what makes a font unique
#    vs. its near neighbors) but lets the model see the neighbors' full
#    profiles directly instead of us hand-summarizing them into a tag list,
#    and generating several fonts in one shared context should discourage
#    the same-template-every-time collapse that isolated per-font calls are
#    prone to. A random per-conversation "voice" hint (STYLE_HINTS) adds a
#    second, ACROSS-conversation diversity axis on top of that.
#
# Failure criterion (retried) is malformed JSON / wrong shape ONLY -- a font
# that comes back with fewer than queriesPerFont queries is NOT a failure,
# see parseOutput.

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix

from utils import Description, loadDescriptionsFromSource, Config

load_dotenv()  # picks up UNSLOTH_API_KEY from a gitignored .env in the repo root

UNSLOTH_MODEL = "unsloth/gemma-4-E4B-it-qat-GGUF"
# Unsloth Desktop's docs say it boots on 8888 or 8000 depending on version/
# config; try both rather than hardcoding one.
UNSLOTH_PORTS = [8888, 8000]

# Fonts bundled into a single conversation (see groupFonts). Keep
# GROUP_SIZE * TOKENS_PER_FONT (+ a few hundred for the prompt) comfortably
# under the app's 8192-token context.
GROUP_SIZE = 5
QUERIES_PER_FONT = 16
# Max query length dropped to 12 words (from 28) -- these are font search
# queries, not descriptions, so the per-font token need shrinks a lot: 16
# queries at up to ~12 words each is ~16*16=256 tokens of raw content. This
# still leaves real headroom above that estimate.
TOKENS_PER_FONT = 500
# Length scaffold bounds (see buildLengthScaffold) -- shortest and longest
# query, in words, across the whole per-font list.
MIN_QUERY_WORDS = 3
MAX_QUERY_WORDS = 12

# Concurrent in-flight requests to the local server. Tested at 2 -- no real
# wall-clock benefit (the server queues requests rather than batching them),
# just extra VRAM pressure for nothing. Sequential is the confirmed-good
# setting.
MAX_CONCURRENCY = 4
# Groups between incremental saves, so a crash mid-run during a long
# unattended pass doesn't lose everything back to the last full save.
CHECKPOINT_EVERY = 50

# Randomly injected "voice" per conversation -- a cheap way to break a
# model's habitual sentence template, adding diversity ACROSS conversations
# on top of the WITHIN-conversation diversity the grouping forces.
STYLE_HINTS = [
    "Write like a busy developer typing quick, slightly informal search terms.",
    "Write like a professional graphic designer browsing a type foundry's catalog.",
    "Write like someone describing the font out loud to a colleague who can't see it.",
    "Write a mix of terse keyword-style fragments and full descriptive sentences.",
    "Write like a typography student explaining what's distinctive about the letterforms.",
]

def buildLengthScaffold(queriesPerFont: int, rng: random.Random,
                         minWords: int = MIN_QUERY_WORDS, maxWords: int = MAX_QUERY_WORDS) -> list[int]:
    """
    A per-slot target word count spread evenly across [minWords, maxWords],
    order shuffled. Prose length-band quotas ("6 of these should be long")
    got ignored entirely by the model in practice even though a neighboring
    exact-count instruction was followed almost perfectly -- an enumerated
    per-slot target is a mechanical checklist instead of something to skim
    past, matching the count instruction's own numbered-item framing.
    """
    if queriesPerFont <= 1:
        return [minWords] * max(queriesPerFont, 0)
    step = (maxWords - minWords) / (queriesPerFont - 1)
    targets = [round(minWords + i * step) for i in range(queriesPerFont)]
    rng.shuffle(targets)
    return targets


def buildSystemPrompt(queriesPerFont: int, lengthTargets: list[int]) -> str:
    """
    NOTE: the calibration examples below are hand-written to actually BE
    MIN_QUERY_WORDS/MAX_QUERY_WORDS words -- if those constants change,
    rewrite the examples to match (word-count them by hand, don't guess).
    Two rounds of "~N words" instruction alone (a prose quota, then this
    same per-slot scaffold without examples) got ignored: the model followed
    the exact-count instruction almost perfectly but every query still
    landed in a narrow 6-9 word band regardless of target. It can count
    LIST ITEMS reliably but doesn't reliably count WORDS while writing one --
    a concrete example at each extreme, labeled with its real count, gives
    it something to pattern-match instead of an abstract number to estimate.
    """
    scaffold = "\n".join(f"{i + 1}. ~{words} words" for i, words in enumerate(lengthTargets))

    return f"""You are a font search expert. You will be shown several fonts that are visually similar to each other. For EACH font, generate realistic search queries a designer might type to find exactly that font and not the others in this group.

For calibration, here is what real queries at each extreme actually look like (for an unrelated example font, a rounded slab serif) -- count the words yourself, these are exact:
- {MIN_QUERY_WORDS} words: "playful rounded slab-serif"
- {MAX_QUERY_WORDS} words: "a heavy rounded slab serif with thick strokes ideal for playful headlines"
Match those real lengths, not something safely in between.

Each font's list of queries MUST contain EXACTLY {queriesPerFont} items, in this exact order, each one close to its target length (a word or two over/under is fine, but stay close -- these are short search-bar queries, never more than {MAX_QUERY_WORDS} words):
{scaffold}

Queries must:
- Describe visual qualities that distinguish this font from the others shown here -- stroke contrast, terminal style, weight distribution, x-height, historical period, construction geometry, emotional register
- Sound like natural search queries, not academic descriptions
- Never mention the font name or its number
- Never mention support for specific characters

Critically: vary the SENTENCE STRUCTURE and OPENING WORDS between different fonts in this group, not just across the length slots within one font. Do not reuse the same sentence template for every font -- if font 1's queries all start "A geometric sans-serif...", font 2's queries should not follow that same shape.

Return ONLY a JSON object mapping each font's number (as a string) to its list of exactly {queriesPerFont} queries. No explanation, no markdown, no preamble. Example for 2 fonts (shortened to 3 queries each here for brevity -- yours must have {queriesPerFont} each):
{{"1": ["...", "...", "..."], "2": ["...", "...", "..."]}}"""


def connectUnsloth() -> OpenAI:
    apiKey = os.environ.get("UNSLOTH_API_KEY")
    if not apiKey:
        raise RuntimeError("Set the UNSLOTH_API_KEY environment variable to your sk-unsloth-... key.")

    lastError = None
    for port in UNSLOTH_PORTS:
        client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key=apiKey, timeout=600.0)
        try:
            models = client.models.list()
            ids = [m.id for m in models.data]
            print(f"Connected to Unsloth Desktop on port {port}. Loaded models: {ids}")
            return client
        except Exception as e:
            lastError = e
    raise RuntimeError(f"Could not reach Unsloth Desktop on ports {UNSLOTH_PORTS}: {lastError}")


def buildFontVectors(descriptions: dict[str, Description]):
    """Sparse (font x vocab) weighted tag/adjective matrix, L2-normalized so
    a dot product is cosine similarity -- the same signal generate.py's
    tagOverlap used, vectorized so grouping the whole corpus is tractable."""
    vocab = {}
    for desc in descriptions.values():
        for tag in list(desc.tags.keys()) + list(desc.adjectives):
            vocab.setdefault(tag, len(vocab))

    names = list(descriptions.keys())
    matrix = lil_matrix((len(names), len(vocab)), dtype=np.float32)
    for row, name in enumerate(names):
        desc = descriptions[name]
        for tag, weight in desc.tags.items():
            matrix[row, vocab[tag]] = weight
        for adjective in desc.adjectives:
            col = vocab[adjective]
            matrix[row, col] = max(matrix[row, col], 1.0)

    matrix = matrix.tocsr()
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1))).reshape(-1)
    norms[norms == 0] = 1.0
    matrix = matrix.multiply(1.0 / norms[:, None]).tocsr()
    return names, matrix


def groupFonts(descriptions: dict[str, Description], groupSize: int, neighborPool: int = 40, seed: int = 0):
    """
    Greedily bundle fonts into groups of `groupSize` mutually-similar fonts,
    by tag/adjective cosine similarity. Not a strict/optimal clustering --
    pop a font, take its still-ungrouped nearest neighbors, repeat; good
    enough for prompt construction and cheap at corpus scale.
    """
    if len(descriptions) <= groupSize:
        return [list(descriptions.keys())] if descriptions else []

    names, matrix = buildFontVectors(descriptions)
    nameIndex = {name: i for i, name in enumerate(names)}

    neighborModel = NearestNeighbors(n_neighbors=min(neighborPool, len(names)), metric="cosine")
    neighborModel.fit(matrix)
    _, neighborIdx = neighborModel.kneighbors(matrix)

    rng = random.Random(seed)
    remaining = set(names)
    order = names.copy()
    rng.shuffle(order)

    groups = []
    for name in order:
        if name not in remaining:
            continue
        remaining.discard(name)
        group = [name]

        for neighborRow in neighborIdx[nameIndex[name]]:
            if len(group) >= groupSize:
                break
            candidate = names[neighborRow]
            if candidate in remaining:
                group.append(candidate)
                remaining.discard(candidate)

        if len(group) < groupSize and remaining:
            fillers = rng.sample(sorted(remaining), min(groupSize - len(group), len(remaining)))
            for filler in fillers:
                group.append(filler)
                remaining.discard(filler)

        groups.append(group)

    return groups


def buildFontBlock(index: int, description: Description) -> str:
    parts = [f"Font {index}:"]
    if description.adjectives:
        parts.append("  Adjectives: " + ", ".join(description.adjectives))
    if description.tags:
        strong = [t for t, w in description.tags.items() if w > 0.3]
        weak = [t for t, w in description.tags.items() if w <= 0.3]
        if strong:
            parts.append("  Strong style tags: " + ", ".join(strong))
        if weak:
            parts.append("  Possible style tags: " + ", ".join(weak))
    if description.plainText:
        parts.append(f"  Description: {description.plainText}")
    return "\n".join(parts)


def extractJsonObjects(text: str) -> list[str]:
    """
    All balanced top-level {...} substrings in text, in order of
    appearance, tracking string/escape state so braces inside quoted
    strings don't confuse the count. Some models (observed on Gemma-4-E4B,
    which ignores the Qwen-specific enable_thinking kwarg entirely) emit
    plain free-text reasoning before the JSON with no delimiter at all --
    not wrapped in <think> tags, so a leading-<think>-strip does nothing.
    Returning every candidate (not just the first) matters because that
    reasoning prose could itself contain a balanced brace pair before the
    real payload -- parseOutput tries each in order and takes the first
    that actually validates, rather than trusting the first one found.
    """
    candidates = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        inString = False
        escaped = False
        start = i
        j = i
        closed = False
        while j < n:
            ch = text[j]
            if inString:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    inString = False
            else:
                if ch == '"':
                    inString = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start:j + 1])
                        closed = True
                        break
            j += 1
        i = (j + 1) if closed else (start + 1)
    return candidates


def parseOutput(text: str, expectedKeys: list[str]) -> dict[str, list[str]] | None:
    """
    Only structural/formatting problems count as failure: invalid JSON,
    wrong top-level type, or a missing/non-list-of-strings entry for one of
    the group's fonts. A font coming back with fewer than queriesPerFont
    queries (model ran out of things to say, or truncated early) is fine and
    is NOT treated as a failure -- whatever queries it produced are kept.
    """
    for candidate in extractJsonObjects(text.strip()):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict) or not all(key in parsed for key in expectedKeys):
            continue
        if not all(isinstance(parsed[key], list) and all(isinstance(q, str) for q in parsed[key]) for key in expectedKeys):
            continue
        return parsed
    return None


def generateGroup(
    client: OpenAI,
    group: list[str],
    descriptions: dict[str, Description],
    queriesPerFont: int,
    rng: random.Random,
) -> dict[str, list[str] | None]:
    blocks = [buildFontBlock(i + 1, descriptions[name]) for i, name in enumerate(group)]
    styleHint = rng.choice(STYLE_HINTS)
    lengthTargets = buildLengthScaffold(queriesPerFont, rng)
    userContent = (
        f"{styleHint}\n\n"
        f"Generate EXACTLY {queriesPerFont} search queries for EACH of the following {len(group)} fonts "
        f"(not fewer -- {queriesPerFont} per font, following the per-slot length scaffold from the system prompt):\n\n"
        + "\n\n".join(blocks)
    )
    messages = [
        {"role": "system", "content": buildSystemPrompt(queriesPerFont, lengthTargets)},
        {"role": "user", "content": userContent},
        # Assistant-turn prefill: a trailing assistant message makes the server
        # continue generation from this partial content instead of starting a
        # fresh turn, which structurally rules out free-text reasoning before
        # the JSON. Needed because enable_thinking=False is Qwen-specific and
        # some models (observed on Gemma-4-E4B) ignore it entirely, emitting
        # plain prose before the JSON with no delimiter to strip. Most
        # OpenAI-compatible servers return only the new continuation when the
        # last message is assistant-role, so the prefix is re-added below.
        {"role": "assistant", "content": "{"},
    ]
    expectedKeys = [str(i + 1) for i in range(len(group))]

    response = client.chat.completions.create(
        model=UNSLOTH_MODEL,
        messages=messages,
        max_tokens=TOKENS_PER_FONT * len(group),
        temperature=rng.uniform(0.6, 1.0),
        top_p=0.95,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    # Whether the server echoes the assistant prefill back or returns only
    # the new continuation varies by implementation -- handle both instead
    # of assuming one.
    content = (response.choices[0].message.content or "").lstrip()
    text = content if content.startswith("{") else "{" + content
    parsed = parseOutput(text, expectedKeys)

    def resultFor(key: str):
        if parsed is None:
            return None
        queries = parsed[key]
        # Fewer than queriesPerFont is fine by design (see parseOutput), but
        # zero is different: an empty list still passes the "list of
        # strings" check (vacuously -- all() over [] is True), so without
        # this it silently saves as a "successful" font with no actual
        # queries, which then crashes anything downstream that samples from
        # it (e.g. trainTextMLP.py's rng.integers(0)). Treat empty as a real
        # failure so it gets retried instead of poisoning the output.
        return queries if queries else None

    return {name: resultFor(str(i + 1)) for i, name in enumerate(group)}


def saveResults(output: str, completed: dict):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(completed, f, indent=2)


def generate(
    descriptions: dict[str, Description],
    output: str = os.path.join("results", "fontQueriesV2.json"),
    queriesPerFont: int = QUERIES_PER_FONT,
    groupSize: int = GROUP_SIZE,
    maxRetries: int = 2,
    seed: int = 0,
    maxConcurrency: int = MAX_CONCURRENCY,
):
    client = connectUnsloth()
    rng = random.Random(seed)

    completed = {}
    if os.path.exists(output):
        with open(output) as f:
            loaded = json.load(f)
        # Empty-list entries could have been saved by earlier runs before
        # generateGroup started treating zero queries as a failure -- don't
        # let those count as "done", or they'd never get regenerated.
        completed = {k: v for k, v in loaded.items() if v}
        droppedEmpty = len(loaded) - len(completed)
        print(f"Resuming -- {len(completed)} already done" +
              (f" ({droppedEmpty} empty entries dropped for retry)" if droppedEmpty else ""))

    pending = {k: v for k, v in descriptions.items() if k not in completed}
    print(f"Grouping {len(pending)} fonts by style similarity...")
    groups = groupFonts(pending, groupSize, seed=seed)
    print(f"{len(groups)} groups of up to {groupSize} fonts, concurrency={maxConcurrency}")

    try:
        for attempt in range(maxRetries + 1):
            if not groups:
                break
            if attempt > 0:
                print(f"Retry pass {attempt} for {sum(len(g) for g in groups)} failed fonts...")

            stillFailing = []
            doneCount = 0
            # Per-group seeds drawn up front from the single-threaded rng --
            # random.Random isn't documented thread-safe, so each worker gets
            # its own private instance instead of sharing one across threads.
            groupSeeds = [rng.randrange(2 ** 30) for _ in groups]

            with ThreadPoolExecutor(max_workers=maxConcurrency) as executor:
                futures = {
                    executor.submit(generateGroup, client, group, descriptions, queriesPerFont, random.Random(groupSeed)): group
                    for group, groupSeed in zip(groups, groupSeeds)
                }
                for future in as_completed(futures):
                    group = futures[future]
                    try:
                        results = future.result()
                    except Exception as e:
                        print(f"\n  request failed for group {group}: {e}")
                        results = {name: None for name in group}

                    failed = [name for name in group if results[name] is None]
                    if failed:
                        stillFailing.extend(failed)
                    for name in group:
                        if results[name] is not None:
                            completed[name] = results[name]

                    doneCount += 1
                    print(f"\r  [{doneCount}/{len(groups)} groups]", end="")
                    if doneCount % CHECKPOINT_EVERY == 0:
                        saveResults(output, completed)

            print()
            failedDescriptions = {name: descriptions[name] for name in stillFailing}
            groups = groupFonts(failedDescriptions, groupSize, seed=seed) if failedDescriptions else []

    except KeyboardInterrupt:
        pass
    finally:
        saveResults(output, completed)
        stillFailingCount = sum(len(g) for g in groups)
        print(f"Saved {len(completed)} entries. Still failing: {stillFailingCount}")


if __name__ == "__main__":
    config = Config().load(os.path.join("configs", "vit.json"))
    descriptions = loadDescriptionsFromSource(config.dataset)
    generate(descriptions)
