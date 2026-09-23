"""SL/EN parity and integration checks for agent/static_prompt.py.

Run from the repo root:  python3 -m unittest discover -s tests -v

stdlib only: agent.static_prompt and agent.company_prompt have no livekit
imports, so this runs without the worker's runtime dependencies. worker.py
and tools.py do import livekit, so they are checked at the source level.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from agent import static_prompt as sp
from agent.company_prompt import render_company_prompt

REPO = Path(__file__).resolve().parent.parent

SOURCES = {
    "sl": sp._STATIC_PROMPT_SL_SOURCE,
    "en": sp._STATIC_PROMPT_EN_SOURCE,
}
RENDERED = {
    "sl": sp.STATIC_PROMPT_SL,
    "en": sp.STATIC_PROMPT_EN,
}


def _ids_by_section(source: str) -> dict[str, str]:
    """rule ID -> the section header it sits under."""
    header_re = re.compile(r"^(%s):$" % "|".join(re.escape(h) for h in sp.SECTIONS), re.M)
    result = {}
    current = None
    for line in source.splitlines():
        m = header_re.match(line)
        if m:
            current = m.group(1)
            continue
        for rule_id in sp.rule_ids(line):
            result[rule_id] = current
    return result


class RuleIdParity(unittest.TestCase):
    def test_ids_are_unique_per_language(self):
        for lang, src in SOURCES.items():
            ids = sp.rule_ids(src)
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            self.assertEqual(dupes, [], f"{lang}: duplicate rule IDs")

    def test_every_sl_rule_exists_in_en(self):
        sl = set(sp.rule_ids(SOURCES["sl"]))
        en = set(sp.rule_ids(SOURCES["en"]))
        missing = sorted(sl - en - sp.SL_ONLY_RULE_IDS)
        self.assertEqual(
            missing,
            [],
            "SL rules missing from STATIC_PROMPT_EN — add them to EN, or to "
            "SL_ONLY_RULE_IDS if they are genuinely Slovenian-specific",
        )

    def test_every_en_rule_exists_in_sl(self):
        sl = set(sp.rule_ids(SOURCES["sl"]))
        en = set(sp.rule_ids(SOURCES["en"]))
        missing = sorted(en - sl - sp.EN_ONLY_RULE_IDS)
        self.assertEqual(
            missing,
            [],
            "EN rules missing from STATIC_PROMPT_SL — add them to SL, or to "
            "EN_ONLY_RULE_IDS with a reason",
        )

    def test_allowlists_are_not_stale(self):
        sl = set(sp.rule_ids(SOURCES["sl"]))
        en = set(sp.rule_ids(SOURCES["en"]))
        self.assertEqual(sorted(sp.SL_ONLY_RULE_IDS - sl), [], "SL_ONLY lists IDs not in SL")
        self.assertEqual(sorted(sp.SL_ONLY_RULE_IDS & en), [], "SL_ONLY IDs also exist in EN")
        self.assertEqual(sorted(sp.EN_ONLY_RULE_IDS - en), [], "EN_ONLY lists IDs not in EN")
        self.assertEqual(sorted(sp.EN_ONLY_RULE_IDS & sl), [], "EN_ONLY IDs also exist in SL")

    def test_sl_prefix_matches_allowlist(self):
        sl = set(sp.rule_ids(SOURCES["sl"]))
        prefixed = {i for i in sl if i.startswith("R-SL-")}
        self.assertEqual(sorted(prefixed ^ sp.SL_ONLY_RULE_IDS), [])

    def test_sections_in_order(self):
        for lang, src in SOURCES.items():
            headers = re.findall(r"^([A-Z][A-Z &]+):$", src, re.M)
            self.assertEqual(tuple(headers), sp.SECTIONS, f"{lang}: section headers")

    def test_shared_rules_sit_in_the_same_section(self):
        sl = _ids_by_section(SOURCES["sl"])
        en = _ids_by_section(SOURCES["en"])
        moved = sorted(
            f"{i}: SL={sl[i]} EN={en[i]}" for i in sl.keys() & en.keys() if sl[i] != en[i]
        )
        self.assertEqual(moved, [])

    def test_every_rule_is_inside_a_section(self):
        for lang, src in SOURCES.items():
            orphans = sorted(i for i, sec in _ids_by_section(src).items() if sec is None)
            self.assertEqual(orphans, [], f"{lang}: rules above the first section")


class RenderedPrompt(unittest.TestCase):
    def test_no_rule_tags_or_id_references_reach_the_model(self):
        # A bare "R-FOO" in prose would point at a tag the model never sees.
        for lang, text in RENDERED.items():
            self.assertEqual(re.findall(r"\bR-[A-Z]+(?:-[A-Z0-9]+)*\b", text), [], lang)

    def test_no_incident_forensics_in_runtime_string(self):
        pattern = re.compile(
            r"\b20\d\d-\d\d-\d\d\b|observed|measured|reverted|OB-0000(?!37)",
            re.IGNORECASE,
        )
        for lang, text in RENDERED.items():
            self.assertEqual(pattern.findall(text), [], lang)


# Minimal booking-v2 init payload: two categories, one single-employee
# service, one multi-employee service, two staff sharing a first name.
_INIT = {
    "company": {"naziv": "Test d.o.o.", "multiple_services_online": False},
    "categories": [{"id": "c1", "name": "Masaže"}, {"id": "c2", "name": "Nega"}],
    "servicesByCategory": {
        "c1": [{"id": "s1", "naziv": "Masaža glave", "cena": 20, "trajanjeMin": 30}],
        "c2": [{"id": "s2", "naziv": "Nega obraza", "cena": 40, "trajanjeMin": 60}],
    },
    "services": [],
    "employees_ui": [
        {"id": "e1", "label": "Luka Dobrovoljec"},
        {"id": "e2", "label": "Maja Hribar"},
        {"id": "e3", "label": "Maja Novak"},
    ],
    "employeesByServiceId": {"s1": ["e1"], "s2": ["e1", "e2", "e3"]},
}

# Literal markers each static prompt tells the model to look for in the
# company block. If company_prompt.py rewords one, the rule silently stops
# pointing at anything — this catches that.
_COMPANY_MARKERS = {
    "sl": ['"opravljajo: ..."', '"opravlja SAMO ..."', '"v pogovoru:"', '"SAMO ..."'],
    "en": ['"performed by: ..."', "is the ONLY staff member for this service", '"say:"'],
}


def _marker_text(marker: str) -> str:
    return marker.strip('"').replace(" ...", "").rstrip()


class CompanyPromptIntegration(unittest.TestCase):
    def test_markers_quoted_in_static_prompt_exist_in_company_block(self):
        for lang, markers in _COMPANY_MARKERS.items():
            block = render_company_prompt(_INIT, lang)
            prompt = " ".join(RENDERED[lang].split())
            for marker in markers:
                self.assertTrue(marker in prompt, f"{lang}: static prompt no longer quotes {marker}")
                self.assertTrue(_marker_text(marker) in block, f"{lang}: company block lacks {marker}")

    def test_company_block_does_not_contradict_employee_preference_rule(self):
        # 26c14ca: the company block once said "if the caller doesn't say,
        # book anyone", which won on recency over the ask-first rule.
        sl = render_company_prompt(_INIT, "sl")
        en = render_company_prompt(_INIT, "en")
        self.assertIn("PREDEN vprašaš za dan", sl)
        self.assertIn("BEFORE asking about the day", en)
        for text in (sp.STATIC_PROMPT_SL, sp.STATIC_PROMPT_EN):
            self.assertNotIn("never mentions an employee at all", text)

    def test_shared_first_name_gets_full_label(self):
        block = render_company_prompt(_INIT, "sl")
        self.assertIn("v pogovoru: Luka\n", block + "\n")
        self.assertIn("v pogovoru: Maja Hribar", block)


class ToolAndDateIntegration(unittest.TestCase):
    """Source-level: worker.py and tools.py import livekit."""

    tools_src = (REPO / "agent" / "tools.py").read_text()
    worker_src = (REPO / "agent" / "worker.py").read_text()

    def test_tool_result_fields_named_in_prompt_are_produced(self):
        for field in ("time_of_day", "requiresPayment", "must_check_slots_first"):
            for lang, text in RENDERED.items():
                self.assertIn(field, text, f"{lang} prompt no longer mentions {field}")
        for field in ('"time_of_day"', '"day_labels"', '"day_label"', "must_check_slots_first"):
            self.assertIn(field, self.tools_src, f"tools.py no longer produces {field}")

    def test_date_context_references_tool_labels(self):
        self.assertIn("day_labels", self.worker_src)
        self.assertIn("day_label", self.worker_src)

    def test_worker_uses_this_module(self):
        self.assertIn(
            "from agent.static_prompt import STATIC_PROMPT_EN, STATIC_PROMPT_SL",
            self.worker_src,
        )
        self.assertNotIn('STATIC_PROMPT_SL = """', self.worker_src)
        self.assertIn("static_prompt\n        + \"\\n\\n\"\n        + _build_date_context", self.worker_src)


if __name__ == "__main__":
    unittest.main()
