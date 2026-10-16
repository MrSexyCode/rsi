"""Prompt adapters: one per eval set, versioned, each a fixed template text plus per-item fields.

Every prompt ends with the training environment's answer instruction (Omanic):

    Reason step by step. Then end your response with exactly this line:
    Answer: <...>

so one extraction rule (last "Answer:" line after the last </think>, see scoring.extract_answer) serves every set.
Only the placeholder inside <...> differs per family.

Rendering substitutes {field} placeholders in a single regex pass, so text inside a field (untrusted context
paragraphs) is never re-scanned for placeholders. Every rendered prompt must equal prompt.strip(): Pi strips
whitespace from the prompt it reads on stdin, so a prompt with surrounding whitespace would differ between the
direct and the Pi path.

Bump an adapter's `version` whenever its template or field formatting changes; the manifest records the version,
the exact template text and its sha256.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
from dataclasses import dataclass
from functools import lru_cache

from harness_rl.core.env_paths import PLAYGROUND

ANSWER_INSTRUCTION = "Reason step by step. Then end your response with exactly this line:"

SHORT_ANSWER_LINE = "Answer: <the short final answer only: an entity, date, or number>"
OMANIC_ANSWER_LINE = "Answer: <the letter of the correct option>"
BABILONG_ANSWER_LINE = "Answer: <your answer, in the format given above>"
RULER_VT_ANSWER_LINE = "Answer: <the names of all the variables, separated by spaces>"

_FIELD_RE = re.compile(r"\{(\w+)\}")


def answer_suffix(answer_line: str) -> str:
    return f"{ANSWER_INSTRUCTION}\n{answer_line}"


def check_prompt(prompt: str, answer_line: str) -> None:
    """Hard requirements on every eval prompt."""
    if prompt != prompt.strip():
        raise ValueError("prompt must equal prompt.strip() (Pi strips stdin whitespace)")
    if not prompt.endswith(answer_suffix(answer_line)):
        raise ValueError("prompt must end with the answer-format instruction")


@dataclass(frozen=True)
class PromptAdapter:
    name: str
    version: str
    template: str
    answer_line: str
    description: str = ""

    @property
    def fields(self) -> list[str]:
        return sorted(set(_FIELD_RE.findall(self.template)))

    @property
    def template_sha256(self) -> str:
        return hashlib.sha256(self.template.encode()).hexdigest()

    def render(self, **fields: str) -> str:
        if sorted(fields) != self.fields:
            raise KeyError(f"{self.name}: fields {sorted(fields)} != template fields {self.fields}")
        prompt = _FIELD_RE.sub(lambda m: fields[m.group(1)], self.template)
        check_prompt(prompt, self.answer_line)
        return prompt

    def manifest(self) -> dict:
        return {"name": self.name, "version": self.version, "template": self.template,
                "template_sha256": self.template_sha256, "fields": self.fields, "answer_line": self.answer_line,
                "description": self.description}


def _fill(template: str, **consts: str) -> str:
    """Fill some placeholders now (single pass), leave the others for render()."""
    return _FIELD_RE.sub(lambda m: consts.get(m.group(1), m.group(0)), template)


# ---------------------------------------------------------------- Omanic (closed-book, training distribution)

@lru_cache(maxsize=1)
def omanic_preprocess():
    """playground/omanic/preprocess.py imported by path (read-only; only its constants are used)."""
    path = PLAYGROUND / "omanic/preprocess.py"
    spec = importlib.util.spec_from_file_location("omanic_preprocess", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class OmanicAdapter(PromptAdapter):
    """Passthrough: the prompt stored in test_shuffled.parquet is used verbatim. render() only checks that it is an
    instance of the Omanic multi-hop template (whose exact text is recorded as `template`)."""

    def render(self, *, prompt: str) -> str:  # type: ignore[override]
        rx = re.escape(self.template).replace(re.escape("{question}"), "(?P<q>.+?)").replace(
            re.escape("{options}"), "(?P<o>A\\. [^\\n]*\\nB\\. [^\\n]*\\nC\\. [^\\n]*\\nD\\. [^\\n]*)")
        if not re.fullmatch(rx, prompt, flags=re.S):  # re.S lets the question span lines; options are one line each
            raise ValueError("Omanic prompt does not match the Omanic template")
        check_prompt(prompt, self.answer_line)
        return prompt


def _omanic_adapter() -> OmanicAdapter:
    pre = omanic_preprocess()
    template = _fill(pre.INSTRUCTION, intro=pre.MULTI_HOP_INTRO)
    return OmanicAdapter("omanic_mc", "1", template, OMANIC_ANSWER_LINE,
                         "Verbatim Omanic prompt from playground/omanic/preprocess.py (multi-hop intro).")


# ---------------------------------------------------------------- in-context multi-hop QA

QA_TEMPLATE = (
    "Answer the following question using the paragraphs below. Answering it requires several reasoning steps.\n"
    "\n"
    "{paragraphs}\n"
    "\n"
    "Question: {question}\n"
    "\n" + answer_suffix(SHORT_ANSWER_LINE)
)


def format_paragraphs(paragraphs: list[tuple[str, str]]) -> str:
    """Numbered like playground/musique/preprocess.py: "[P1] Title\\ntext", blank line between paragraphs."""
    return "\n\n".join(f"[P{i + 1}] {title.strip()}\n{text.strip()}" for i, (title, text) in enumerate(paragraphs))


def _qa_adapter(name: str, source: str) -> PromptAdapter:
    return PromptAdapter(name, "1", QA_TEMPLATE, SHORT_ANSWER_LINE,
                         f"{source}: all context paragraphs in source order, numbered [P1].., then the question.")


# ---------------------------------------------------------------- BABILong

# Copied verbatim from booydar/babilong@7a6efee29f5cac03c3c410e6799c80fd2ffe3610 babilong/prompts.py
# (TASK_TEMPLATE, USER_TEMPLATE, DEFAULT_PROMPTS['qa1'..'qa5']); tests/test_evals.py re-checks them against the
# downloaded file with ast.literal_eval (no code from the download is executed).
BABILONG_COMMIT = "7a6efee29f5cac03c3c410e6799c80fd2ffe3610"
BABILONG_TASK_TEMPLATE = '{instruction}\n\n{examples}\n\n{post_prompt}'
BABILONG_USER_TEMPLATE = '<context>\n{context}\n</context>\n\nQuestion: {question}'
BABILONG_DEFAULT_TEMPLATE = f'{BABILONG_TASK_TEMPLATE}\n\n{BABILONG_USER_TEMPLATE}'
BABILONG_PROMPTS = {
    'qa1': {
        'instruction': 'I will give you context with the facts about positions of different persons hidden in some random text and a question. You need to answer the question based only on the information from the facts. If a person was in different locations, use the latest location to answer the question.',
        'examples': '<example>\nCharlie went to the hallway. Judith come back to the kitchen. Charlie travelled to balcony. Where is Charlie?\nAnswer: The most recent location of Charlie is balcony.\n</example>\n\n<example>\nAlan moved to the garage. Charlie went to the beach. Alan went to the shop. Rouse travelled to balcony. Where is Alan?\nAnswer: The most recent location of Alan is shop.\n</example>',
        'post_prompt': 'Always return your answer in the following format: The most recent location of ’person’ is ’location’. Do not write anything else after that.',
    },
    'qa2': {
        'instruction': 'I give you context with the facts about locations and actions of different persons hidden in some random text and a question.You need to answer the question based only on the information from the facts.\nIf a person got an item in the first location and travelled to the second location the item is also in the second location. If a person dropped an item in the first location and moved to the second location the item remains in the first location.',
        'examples': '<example>\nCharlie went to the kitchen. Charlie got a bottle. Charlie moved to the balcony. Where is the bottle?\nAnswer: The bottle is in the balcony.\n</example>\n<example>\nAlan moved to the garage. Alan got a screw driver. Alan moved to the kitchen. Where is the screw driver?\nAnswer: The screw driver is in the kitchen.\n</example>',
        'post_prompt': 'Always return your answer in the following format: The ’item’ is in ’location’. Do not write anything else after that.',
    },
    'qa3': {
        'instruction': 'I give you context with the facts about locations and actions of different persons hidden in some random text and a question. You need to answer the question based only on the information from the facts.\nIf a person got an item in the first location and travelled to the second location the item is also in the second location. If a person dropped an item in the first location and moved to the second location the item remains in the first location.',
        'examples': '<example>\nJohn journeyed to the bedroom. Mary grabbed the apple. Mary went back to the bathroom. Daniel journeyed to the bedroom. Daniel moved to the garden. Mary travelled to the kitchen. Where was the apple before the kitchen?\nAnswer: Before the kitchen the apple was in the bathroom.\n</example>\n<example>\nJohn went back to the bedroom. John went back to the garden. John went back to the kitchen. Sandra took the football. Sandra travelled to the garden. Sandra journeyed to the bedroom. Where was the football before the bedroom?\nAnswer: Before the bedroom the football was in the garden.\n</example>',
        'post_prompt': 'Always return your answer in the following format: Before the $location_1$ the $item$ was in the $location_2$. Do not write anything else after that.',
    },
    'qa4': {
        'instruction': 'I will give you context with the facts about different people, their location and actions, hidden in some random text and a question. You need to answer the question based only on the information from the facts.',
        'examples': '<example>\nThe hallway is south of the kitchen. The bedroom is north of the kitchen. What is the kitchen south of?\nAnswer: bedroom\n</example>\n<example>\nThe garden is west of the bedroom. The bedroom is west of the kitchen. What is west of the bedroom?\nAnswer: garden\n</example>',
        'post_prompt': 'Your answer should contain only one word - location. Do not write anything else after that.',
    },
    'qa5': {
        'instruction': 'I will give you context with the facts about locations and their relations hidden in some random text and a question. You need to answer the question based only on the information from the facts.',
        'examples': '<example>\nMary picked up the apple there. Mary gave the apple to Fred. Mary moved to the bedroom. Bill took the milk there. Who did Mary give the apple to?\nAnswer: Fred\n</example>\n<example>\nJeff took the football there. Jeff passed the football to Fred. Jeff got the milk there. Bill travelled to the bedroom. Who gave the football?\nAnswer: Jeff\n</example>\n<example>\nFred picked up the apple there. Fred handed the apple to Bill. Bill journeyed to the bedroom. Jeff went back to the garden. What did Fred give to Bill?\nAnswer: apple\n</example>',
        'post_prompt': 'Your answer should contain only one word. Do not write anything else after that. Do not explain your answer.',
    },
}
for _p in BABILONG_PROMPTS.values():
    assert not any(_FIELD_RE.search(v) for v in _p.values())


def _babilong_adapter(task: str) -> PromptAdapter:
    """BABILong's own prompt (get_formatted_input with DEFAULT_TEMPLATE and the task's instruction, examples and
    post_prompt; context and question stripped, as the official .strip() of the whole input does), then the
    shared answer instruction."""
    p = BABILONG_PROMPTS[task]
    template = _fill(BABILONG_DEFAULT_TEMPLATE, **p) + "\n\n" + answer_suffix(BABILONG_ANSWER_LINE)
    return PromptAdapter(f"babilong_{task}", "1", template, BABILONG_ANSWER_LINE,
                         f"BABILong {task} official prompt (booydar/babilong@{BABILONG_COMMIT[:8]}) + answer line.")


# ---------------------------------------------------------------- RULER variable tracking

# From NVIDIA/RULER@c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a scripts/data/synthetic/constants.py
# TASKS['variable_tracking'] (model template 'base' = "{task_template}").
RULER_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
RULER_VT_TASK_TEMPLATE = ("Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
                          "{context}\nQuestion: Find all variables that are assigned the value {query} in the text above.")
RULER_VT_ANSWER_PREFIX = (" Answer: According to the chain(s) of variable assignment in the text above, "
                          "{num_v} variables are assigned the value {query}, they are: ")

# RULER prepends one in-context example (its own task template + answer prefix + answer) separated by a newline:
# input = icl_example + '\n' + task_input. The answer prefix of the actual query (which RULER feeds as the start of
# the model's output) is dropped: with a chat model in thinking mode the response cannot be prefilled, and the
# shared answer instruction takes its place.
RULER_VT_TEMPLATE = "{icl_example}\n\n" + RULER_VT_TASK_TEMPLATE + "\n\n" + answer_suffix(RULER_VT_ANSWER_LINE)


def _ruler_vt_adapter() -> PromptAdapter:
    return PromptAdapter("ruler_vt", "1", RULER_VT_TEMPLATE, RULER_VT_ANSWER_LINE,
                         f"RULER variable tracking (NVIDIA/RULER@{RULER_COMMIT[:8]}): one randomized ICL example, "
                         "noise haystack, 1 chain x 4 hops; answer prefix replaced by the shared answer line.")


@lru_cache(maxsize=None)
def get_adapter(name: str) -> PromptAdapter:
    if name == "omanic_mc":
        return _omanic_adapter()
    if name in QA_SOURCES:
        return _qa_adapter(name, QA_SOURCES[name])
    if name.startswith("babilong_") and name[len("babilong_"):] in BABILONG_PROMPTS:
        return _babilong_adapter(name[len("babilong_"):])
    if name == "ruler_vt":
        return _ruler_vt_adapter()
    raise KeyError(name)


QA_SOURCES = {"musique_ctx": "MuSiQue-Ans", "2wiki_ctx": "2WikiMultihopQA", "hotpotqa_ctx": "HotpotQA distractor"}
ADAPTER_NAMES = ["omanic_mc", *QA_SOURCES, *(f"babilong_{t}" for t in BABILONG_PROMPTS), "ruler_vt"]
