"""
Reusable Anki flashcard generator built on top of the generic RAG core.

The card-generation workflow is deliberately different from the interactive
Q&A RAG flow: instead of retrieving chunks and answering one question, it walks
a folder of pre-processed markdown sections (one file = one batch), feeds each
section's full content directly to the LLM, and asks for a batch of
tab-separated Anki cards
(``CardType<TAB>Front<TAB>Back<TAB>Options<TAB>CorrectOption<TAB>Tags``).

Retrieval is still used in a "hybrid" fashion: for each section we pull the most
relevant real quiz/exam questions from a dedicated vector-store collection and
inject them as few-shot calibration examples. A second LLM pass removes
redundant / too-similar / too-long cards before the deck is written to disk.

Besides the plain-text TSV (kept for inspection/debugging), the deck is also
compiled into a native Anki ``.apkg`` via genanki, using a distinct note type
per CardType so each format renders correctly in the real Anki review UI:
MCQ gets a click-to-select note type, FILL_BLANK gets Anki's built-in
type-in-the-answer field, and SHORT_ANSWER gets a plain reveal-based card
(forcing verbatim-typed matching on sentence-length answers would make review
sessions unusable).
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import genanki

from rag_core import GenericRAGChatbot, KnowledgeBaseConfig, PromptConfig

# Fallback format guidance used only when a specific bot does not provide
# its own instructions.
DEFAULT_FORMAT_INSTRUCTIONS = (
    "Output format - one card per line, tab-separated, with EXACTLY six "
    "columns:\n"
    "  CardType<TAB>Front<TAB>Back<TAB>Options<TAB>CorrectOption<TAB>Tags\n"
    "Do not output a header row, bullet points, markdown tables, or any prose "
    "outside the card lines.\n\n"
    "Rules:\n"
    "- One atomic concept per card.\n"
    "- CardType must be one of: MCQ, FILL_BLANK, SHORT_ANSWER.\n"
    "- Front: ask one clear question. For MCQ, do NOT embed the answer "
    "choices in Front -- put them in Options instead.\n"
    "- Back: provide the direct answer with a brief explanation. For MCQ, "
    "do not repeat the option letter (CorrectOption already carries it); "
    "just give the explanation.\n"
    "- Options: ONLY for MCQ, exactly four choices formatted as "
    "'A) ... | B) ... | C) ... | D) ...' (pipe-separated). Leave EMPTY for "
    "FILL_BLANK and SHORT_ANSWER.\n"
    "- CorrectOption: ONLY for MCQ, a single letter (A, B, C, or D) matching "
    "the correct entry in Options. Leave EMPTY for FILL_BLANK and "
    "SHORT_ANSWER.\n"
    "- Tags: space-separated; always include the base tag(s) plus ONE topic tag.\n"
    "- Use real tab characters between columns; do not print the literal string "
    "<TAB>.\n"
)


@dataclass(frozen=True)
class AnkiConfig:
    """Configuration for an Anki deck-generation run."""

    deck_name: str
    output_path: str
    source_md_dir: str
    base_tags: list[str] = field(default_factory=list)
    cards_per_section: int = 15
    quiz_examples_k: int = 5
    run_qc_pass: bool = True
    format_instructions: str = DEFAULT_FORMAT_INSTRUCTIONS
    apkg_output_path: str | None = None


# ----------------------------------------------------------------------
# genanki note types -- one per CardType so each format renders natively
# in the real Anki review UI.
# ----------------------------------------------------------------------
def _stable_id(seed: str) -> int:
    """Deterministic model/deck id so re-generating a deck updates it in
    place instead of creating Anki-side duplicates on every import."""
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (1 << 31)


_BASE_CSS = (
    ".card { font-family: arial; font-size: 18px; text-align: left; "
    "color: black; background-color: white; padding: 12px; }\n"
    ".question { font-weight: bold; margin-bottom: 14px; }\n"
    ".answer, .explanation { margin-top: 12px; }\n"
)

MCQ_MODEL = genanki.Model(
    _stable_id("anki_core:mcq_model:v1"),
    "RAG Chatbot MCQ",
    fields=[
        {"name": "Question"},
        {"name": "OptionA"},
        {"name": "OptionB"},
        {"name": "OptionC"},
        {"name": "OptionD"},
        {"name": "CorrectLetter"},
        {"name": "Explanation"},
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": (
                '<div class="question">{{Question}}</div>\n'
                '<div class="options">\n'
                '  <div class="option" data-letter="A" onclick="ankiMcqSelect(this)">A) {{OptionA}}</div>\n'
                '  <div class="option" data-letter="B" onclick="ankiMcqSelect(this)">B) {{OptionB}}</div>\n'
                '  {{#OptionC}}<div class="option" data-letter="C" onclick="ankiMcqSelect(this)">C) {{OptionC}}</div>{{/OptionC}}\n'
                '  {{#OptionD}}<div class="option" data-letter="D" onclick="ankiMcqSelect(this)">D) {{OptionD}}</div>{{/OptionD}}\n'
                "</div>\n"
                "<script>\n"
                "function ankiMcqSelect(el) {\n"
                "  var options = el.parentElement.querySelectorAll('.option');\n"
                "  options.forEach(function (o) { o.classList.remove('selected'); });\n"
                "  el.classList.add('selected');\n"
                "  el.setAttribute('data-picked', '1');\n"
                "}\n"
                "</script>"
            ),
            "afmt": (
                "{{FrontSide}}\n"
                '<hr id="answer">\n'
                '<div class="explanation">{{Explanation}}</div>\n'
                "<script>\n"
                "(function () {\n"
                '  var correct = "{{CorrectLetter}}";\n'
                "  document.querySelectorAll('.option').forEach(function (o) {\n"
                "    if (o.getAttribute('data-letter') === correct) {\n"
                "      o.classList.add('correct');\n"
                "    } else if (o.getAttribute('data-picked') === '1') {\n"
                "      o.classList.add('incorrect');\n"
                "    }\n"
                "  });\n"
                "})();\n"
                "</script>"
            ),
        }
    ],
    css=(
        _BASE_CSS
        + ".option { padding: 8px 12px; margin: 6px 0; border: 1px solid #ccc; "
        "border-radius: 6px; cursor: pointer; }\n"
        ".option.selected { border-color: #4a90d9; background-color: #eaf2fb; }\n"
        ".option.correct { border-color: #2e7d32; background-color: #c8f7c5; }\n"
        ".option.incorrect { border-color: #c62828; background-color: #f7c5c5; }\n"
    ),
)

FILL_BLANK_MODEL = genanki.Model(
    _stable_id("anki_core:fill_blank_model:v1"),
    "RAG Chatbot Fill In The Blank",
    fields=[{"name": "Question"}, {"name": "Answer"}, {"name": "Explanation"}],
    templates=[
        {
            "name": "Card 1",
            "qfmt": '<div class="question">{{Question}}</div>\n{{type:Answer}}',
            "afmt": (
                '<div class="question">{{Question}}</div>\n'
                "{{type:Answer}}\n"
                '<hr id="answer">\n'
                '<div class="explanation">{{Explanation}}</div>'
            ),
        }
    ],
    css=_BASE_CSS,
)

BASIC_MODEL = genanki.Model(
    _stable_id("anki_core:basic_model:v1"),
    "RAG Chatbot Basic",
    fields=[{"name": "Front"}, {"name": "Back"}],
    templates=[
        {
            "name": "Card 1",
            "qfmt": '<div class="question">{{Front}}</div>',
            "afmt": (
                "{{FrontSide}}\n"
                '<hr id="answer">\n'
                '<div class="answer">{{Back}}</div>'
            ),
        }
    ],
    css=_BASE_CSS,
)


def _parse_mcq_options(options: str) -> dict[str, str]:
    """Parse 'A) foo | B) bar | ...' into {'A': 'foo', 'B': 'bar', ...}."""
    result: dict[str, str] = {}
    for part in options.split("|"):
        match = re.match(r"^([A-D])\)\s*(.+)$", part.strip())
        if match:
            result[match.group(1)] = match.group(2).strip()
    return result


def _split_fill_blank_answer(back: str) -> tuple[str, str]:
    """Split a FILL_BLANK Back field ('term - explanation') into (answer,
    explanation), used to grade the native Anki type-in-the-answer field
    against just the blanked term rather than the whole sentence."""
    if " - " in back:
        answer, explanation = back.split(" - ", 1)
        return answer.strip(), explanation.strip()
    return back.strip(), ""


def build_apkg(rows: list[str], deck_name: str, output_path: str) -> int:
    """Render validated 6-column card rows into a native Anki .apkg deck.

    Returns the number of cards written.
    """
    deck = genanki.Deck(_stable_id(f"deck:{deck_name}"), deck_name)
    count = 0

    for row in rows:
        fields = row.split("\t")
        if len(fields) != 6:
            continue
        card_type, front, back, options, correct, tags = fields
        tag_list = tags.split()

        if card_type == "MCQ":
            choices = _parse_mcq_options(options)
            if correct not in choices:
                print(f"  Skipping MCQ card with unresolved options: {front[:80]!r}")
                continue
            note = genanki.Note(
                model=MCQ_MODEL,
                fields=[
                    front,
                    choices.get("A", ""),
                    choices.get("B", ""),
                    choices.get("C", ""),
                    choices.get("D", ""),
                    correct,
                    back,
                ],
                tags=tag_list,
            )
        elif card_type == "FILL_BLANK":
            answer, explanation = _split_fill_blank_answer(back)
            note = genanki.Note(
                model=FILL_BLANK_MODEL,
                fields=[front, answer, explanation],
                tags=tag_list,
            )
        else:  # SHORT_ANSWER and any unrecognized fallback
            note = genanki.Note(
                model=BASIC_MODEL,
                fields=[front, back],
                tags=tag_list,
            )

        deck.add_note(note)
        count += 1

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    genanki.Package(deck).write_to_file(output_path)
    return count


def _call_with_backoff(fn, *args, max_retries: int = 6, base_delay: float = 5.0, **kwargs):
    """Call fn(*args, **kwargs), retrying on 503/500 with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc)
            if attempt == max_retries - 1:
                raise
            if "503" in msg or "UNAVAILABLE" in msg or "500" in msg or "overloaded" in msg.lower():
                delay = base_delay * (2 ** attempt)
                print(f"  API error (attempt {attempt + 1}/{max_retries}): {msg[:80]}. "
                      f"Retrying in {delay:.0f}s...")
                time.sleep(delay)
            else:
                raise


class AnkiCardBot(GenericRAGChatbot):
    """Generate Anki flashcards from a folder of markdown sections."""

    def __init__(
        self,
        llm_model_name: str,
        embedding_model_name: str,
        api_key: str,
        knowledge_base: KnowledgeBaseConfig,
        anki_config: AnkiConfig,
        prompt_config: PromptConfig | None = None,
    ):
        super().__init__(
            llm_model_name=llm_model_name,
            embedding_model_name=embedding_model_name,
            api_key=api_key,
            knowledge_base=knowledge_base,
            prompt_config=prompt_config,
        )
        self.anki_config = anki_config

    # ------------------------------------------------------------------
    # Source loading
    # ------------------------------------------------------------------
    def _load_section_files(self) -> list[tuple[str, str]]:
        """Return (title, content) for every markdown file in the source dir."""
        source_dir = self.anki_config.source_md_dir
        if not os.path.isdir(source_dir):
            raise FileNotFoundError(
                f"Source markdown directory not found: {source_dir}"
            )

        sections: list[tuple[str, str]] = []
        for md_path in sorted(Path(source_dir).glob("*.md")):
            content = md_path.read_text(encoding="utf-8").strip()
            if content:
                sections.append((md_path.stem, content))

        if not sections:
            raise FileNotFoundError(
                f"No non-empty .md files found in {source_dir}"
            )
        return sections

    # ------------------------------------------------------------------
    # Few-shot example retrieval (hybrid RAG step)
    # ------------------------------------------------------------------
    def _get_example_questions(self, section_text: str) -> str:
        """Retrieve relevant real quiz questions to calibrate the LLM.

        Returns an empty string if the collection is missing or empty so that
        generation can still proceed without examples.
        """
        try:
            results = self.retrieve_docs_only(
                section_text, k=self.anki_config.quiz_examples_k
            )
        except Exception as exc:  # collection missing / not built yet
            print(f"  (no quiz examples retrieved: {exc})")
            return ""

        documents = (results or {}).get("documents") or []
        flattened = [doc for group in documents for doc in group if doc.strip()]
        if not flattened:
            return ""

        return "\n\n".join(f"- {doc.strip()}" for doc in flattened)

    # ------------------------------------------------------------------
    # Prompt construction + generation
    # ------------------------------------------------------------------
    def _build_card_prompt(self, title: str, content: str, examples: str) -> str:
        cfg = self.anki_config
        base_tags = " ".join(cfg.base_tags) if cfg.base_tags else ""
        role = self.prompt_config.assistant_role

        example_block = (
            "Here are real example questions on this subject to match the style "
            "and difficulty (do not copy them verbatim, use them to calibrate):\n"
            f"{examples}\n\n"
            if examples
            else ""
        )

        return (
            f"You are {role}. You are generating Anki cards for an exam that "
            "consists of fill-in-the-blank questions, multiple choice, and short-answer questions.\n\n"
            f"{cfg.format_instructions}\n"
            f"Always include these base tag(s) on every card: {base_tags}\n\n"
            f"{example_block}"
            f"Now generate ~{cfg.cards_per_section} cards from this section "
            f"titled \"{title}\". Pick a single topic tag that fits the section. "
            "Output ONLY the tab-separated card lines.\n\n"
            f"Section content:\n{content}"
        )

    def _generate_cards_for_section(
        self, title: str, content: str, examples: str
    ) -> str:
        prompt = self._build_card_prompt(title, content, examples)
        response = _call_with_backoff(
            self.client.models.generate_content,
            model=self.llm_model_name,
            contents=[prompt],
        )
        return response.text or ""

    # ------------------------------------------------------------------
    # Quality-control / dedup pass
    # ------------------------------------------------------------------
    def _run_qc_pass(self, cards_tsv: str) -> str:
        prompt = (
            "Below is a set of Anki cards, one per line, tab-separated as "
            "CardType<TAB>Front<TAB>Back<TAB>Options<TAB>CorrectOption<TAB>Tags "
            "(Options/CorrectOption are empty except for MCQ rows).\n"
            "Clean this set:\n"
            "- Remove cards that are redundant or too similar to another card.\n"
            "- Remove cards whose front is too long (testing recall of a "
            "paragraph rather than one fact).\n"
            "- Remove trivially easy cards where the answer is embedded in the "
            "front.\n"
            "Keep the rest unchanged. Preserve the exact tab-separated format "
            "and output ONLY the surviving card lines, nothing else.\n\n"
            f"{cards_tsv}"
        )
        response = _call_with_backoff(
            self.client.models.generate_content,
            model=self.llm_model_name,
            contents=[prompt],
        )
        return response.text or ""

    # ------------------------------------------------------------------
    # Validation + orchestration
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_rows(tsv: str) -> list[str]:
        """Keep only well-formed card rows; warn on overly long fronts."""

        def normalize_card_type(value: str) -> str | None:
            normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
            aliases = {
                "MULTIPLE_CHOICE": "MCQ",
                "MCQ": "MCQ",
                "FILL_IN_THE_BLANK": "FILL_BLANK",
                "FILL_IN_BLANK": "FILL_BLANK",
                "FILL_BLANK": "FILL_BLANK",
                "SHORT_ANSWER": "SHORT_ANSWER",
                "SHORTANSWER": "SHORT_ANSWER",
            }
            return aliases.get(normalized)

        def normalize_correct_option(value: str) -> str:
            stripped = value.strip().upper().rstrip(").")
            return stripped if stripped in {"A", "B", "C", "D"} else ""

        def split_embedded_mcq_options(front: str) -> tuple[str, str]:
            """Pull legacy 'question A) ... B) ... C) ... D) ...' choices out
            of a front string into a separate pipe-separated Options field."""
            match = re.search(r"\bA\)\s", front)
            if not match:
                return front, ""
            question = front[: match.start()].strip()
            choices = front[match.start():].strip()
            parts = re.split(r"\s(?=[B-D]\))", choices)
            return question, " | ".join(part.strip() for part in parts)

        def infer_card_type_and_front(front: str, back: str) -> tuple[str | None, str, str]:
            lowered = front.strip().lower()
            if lowered.startswith("mcq:") or lowered.startswith("multiple choice:"):
                return ("MCQ",) + split_embedded_mcq_options(front.split(":", 1)[1].strip())
            if lowered.startswith("fill in the blank:") or lowered.startswith("fill-in-the-blank:"):
                return "FILL_BLANK", front.split(":", 1)[1].strip(), ""
            if lowered.startswith("short answer:"):
                return "SHORT_ANSWER", front.split(":", 1)[1].strip(), ""
            if any(option in front for option in ("A)", "B)", "C)", "D)")):
                return ("MCQ",) + split_embedded_mcq_options(front.strip())
            if "____" in front or "___" in front:
                return "FILL_BLANK", front.strip(), ""
            if len(back.split()) <= 25:
                return "SHORT_ANSWER", front.strip(), ""
            return None, front.strip(), ""

        rows: list[str] = []
        for line in tsv.splitlines():
            line = line.rstrip()
            if not line.strip():
                continue
            # Strip accidental code-fence lines.
            if line.strip().startswith("```"):
                continue

            # Accept model outputs that use literal "<TAB>" instead of actual tabs.
            normalized = line.replace("<TAB>", "\t")
            fields = normalized.split("\t")

            if len(fields) == 6:
                card_type_raw, front, back, options, correct, tags = (
                    field.strip() for field in fields
                )
                card_type = normalize_card_type(card_type_raw)
                if card_type is None:
                    print(f"  Skipping malformed card type: {card_type_raw[:80]!r}")
                    continue
                correct = normalize_correct_option(correct)
            elif len(fields) == 4:
                card_type_raw, front, back, tags = (field.strip() for field in fields)
                card_type = normalize_card_type(card_type_raw)
                if card_type is None:
                    print(f"  Skipping malformed card type: {card_type_raw[:80]!r}")
                    continue
                options, correct = "", ""
                if card_type == "MCQ":
                    front, options = split_embedded_mcq_options(front)
            elif len(fields) == 3:
                front, back, tags = (field.strip() for field in fields)
                card_type, front, options = infer_card_type_and_front(front, back)
                correct = ""
                if card_type is None:
                    print(f"  Skipping malformed card line: {line[:80]!r}")
                    continue
            else:
                print(f"  Skipping malformed card line: {line[:80]!r}")
                continue

            if card_type == "MCQ" and (not options or not correct):
                print(f"  Skipping MCQ card missing options/correct answer: {front[:80]!r}")
                continue

            if len(front.split()) > 25:
                print(f"  Warning: long front (>25 words): {front[:80]!r}")

            rows.append("\t".join([card_type, front, back, options, correct, tags]))
        return rows

    def generate_deck(self) -> dict:
        """Generate the full deck and write it to the configured output path."""
        sections = self._load_section_files()
        print(f"Loaded {len(sections)} section file(s) from "
              f"{self.anki_config.source_md_dir}")

        raw_lines: list[str] = []
        for title, content in sections:
            print(f"\nGenerating cards for section: {title}")
            examples = self._get_example_questions(content)
            cards_tsv = self._generate_cards_for_section(title, content, examples)
            section_rows = self._validate_rows(cards_tsv)
            print(f"  -> {len(section_rows)} valid card(s)")
            raw_lines.extend(section_rows)

        pre_qc_count = len(raw_lines)

        if self.anki_config.run_qc_pass and raw_lines:
            print(f"\nRunning QC/dedup pass over {pre_qc_count} cards...")
            cleaned = self._run_qc_pass("\n".join(raw_lines))
            raw_lines = self._validate_rows(cleaned)
            print(f"  -> {len(raw_lines)} card(s) after QC")

        output_path = self.anki_config.output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(raw_lines))
            if raw_lines:
                handle.write("\n")

        apkg_path = self.anki_config.apkg_output_path or str(
            Path(output_path).with_suffix(".apkg")
        )
        apkg_cards = (
            build_apkg(raw_lines, self.anki_config.deck_name, apkg_path)
            if raw_lines
            else 0
        )

        summary = {
            "sections": len(sections),
            "cards_pre_qc": pre_qc_count,
            "cards_final": len(raw_lines),
            "output_path": output_path,
            "apkg_path": apkg_path,
            "apkg_cards": apkg_cards,
        }
        print(
            f"\nDeck '{self.anki_config.deck_name}' written: "
            f"{summary['cards_final']} cards -> {output_path}\n"
            f"  Anki package: {apkg_cards} card(s) -> {apkg_path}"
        )
        return summary
