"""Typed relations and their templates (plan 1.2; v2 template audit).

Every relation is functional (at most one object per subject). Entity relations map an entity type to an entity
type; attributes map an entity type to an integer (a year or a count) and are only used by comparisons.

Each relation and attribute has a family of key words (`keys`) that names it and nothing else. Every template of a
relation (facts, question phrases) contains one of its own keys and no key of any other relation or attribute
(tests/test_multihop.py checks this), so a fact always visibly states the relation the question asks about
(v1 had paraphrases such as "was trained at" for studied_at that the model did not read as the asked relation).

Templates:
  facts   5 sentence templates with {s} (subject) and {o} (object; {v} for attribute values). Each compiles to a
          regex (fact_regexes) that parses the sentence back; every rendered fact parses to exactly one
          (relation, template) and the original triple.
  nps     2 noun-phrase templates ending in {x}: "the birthplace of {x}". Nested (right-branching) to phrase chains.
  vps     2 relative clauses ending in {y}, about the subject: "whose birthplace is {y}", "who was born in {y}".
          Used by intersections.
  questions (attributes) 2 templates per comparison direction with {a}, {b}.
Parallel relations with the same subject and object types (born_in / lives_in, works_for / studied_at, ...)
exist on purpose: they make "same subject, other relation" distractors real near misses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from harness_rl.adapters.multihop.names import TYPES


@dataclass(frozen=True)
class Relation:
    name: str
    subj: str
    obj: str
    keys: tuple[str, ...]
    facts: tuple[str, ...]
    nps: tuple[str, ...]
    vps: tuple[str, ...]


@dataclass(frozen=True)
class Attribute:
    name: str
    subj: str
    kind: str  # "year" | "count"
    lo: int
    hi: int
    keys: tuple[str, ...]
    facts: tuple[str, ...]
    questions: dict  # "max" | "min" -> templates with {a}, {b}


TYPE_NOUN = {"Person": "person", "City": "city", "Organization": "organization", "Artifact": "artifact",
             "Event": "event"}


def _rel(name, subj, obj, keys, facts, nps, vps) -> Relation:
    return Relation(name, subj, obj, tuple(keys), tuple(facts), tuple(nps), tuple(vps))


RELATIONS: dict[str, Relation] = {r.name: r for r in [
    # ---- Person -> (relative pronoun "who")
    _rel("born_in", "Person", "City", ["born", "birthplace"],
         ["{s} was born in {o}.", "{o} is the birthplace of {s}.", "The birthplace of {s} is {o}.",
          "{s} was born in the city of {o}.", "The city of {o} is the birthplace of {s}."],
         ["the birthplace of {x}", "the city that is the birthplace of {x}"],
         ["whose birthplace is {y}", "who was born in {y}"]),
    _rel("lives_in", "Person", "City", ["lives", "resides", "residence"],
         ["{s} lives in {o}.", "{o} is the city of residence of {s}.", "The city of residence of {s} is {o}.",
          "{s} resides in {o}.", "{s} lives in the city of {o}."],
         ["the city of residence of {x}", "the city that is the residence of {x}"],
         ["whose city of residence is {y}", "who lives in {y}"]),
    _rel("works_for", "Person", "Organization", ["works for", "employer", "employs", "employed"],
         ["{s} works for {o}.", "{o} employs {s}.", "{s} is employed by {o}.", "The employer of {s} is {o}.",
          "{o} is the employer of {s}."],
         ["the employer of {x}", "the organization that employs {x}"],
         ["whose employer is {y}", "who works for {y}"]),
    _rel("studied_at", "Person", "Organization", ["studied at", "alma mater", "graduated from"],
         ["{s} studied at {o}.", "{o} is the alma mater of {s}.", "{s} graduated from {o}.",
          "The alma mater of {s} is {o}.", "{s} studied at and graduated from {o}."],
         ["the alma mater of {x}", "the organization that is the alma mater of {x}"],
         ["whose alma mater is {y}", "who studied at {y}"]),
    _rel("mentored_by", "Person", "Person", ["mentor", "mentored"],
         ["{s} was mentored by {o}.", "{o} mentored {s}.", "The mentor of {s} is {o}.", "{o} is the mentor of {s}.",
          "{o} served as the mentor of {s}."],
         ["the mentor of {x}", "the person who mentored {x}"],
         ["whose mentor is {y}", "who was mentored by {y}"]),
    _rel("debuted_at", "Person", "Event", ["debut", "debuted"],
         ["{s} debuted at {o}.", "{o} featured the debut of {s}.", "The debut of {s} took place at {o}.",
          "{s} made their debut at {o}.", "{o} is the debut event of {s}."],
         ["the debut event of {x}", "the event that featured the debut of {x}"],
         ["whose debut event is {y}", "who debuted at {y}"]),
    # ---- City -> (relative pronoun "that")
    _rel("mayor", "City", "Person", ["mayor"],
         ["The mayor of {s} is {o}.", "{o} is the mayor of {s}.", "{s} is governed by mayor {o}.",
          "{o} serves as the mayor of {s}.", "{s} has {o} as its mayor."],
         ["the mayor of {x}", "the person who is the mayor of {x}"],
         ["whose mayor is {y}", "that is governed by mayor {y}"]),
    _rel("official_partner", "City", "Organization", ["official partner"],
         ["The official partner of {s} is {o}.", "{o} is the official partner of {s}.",
          "{s} has {o} as its official partner.", "{o} serves as the official partner of {s}.",
          "{s} chose {o} as its official partner."],
         ["the official partner of {x}", "the organization that is the official partner of {x}"],
         ["whose official partner is {y}", "that has as its official partner {y}"]),
    _rel("treasure", "City", "Artifact", ["most treasured artifact", "treasures"],
         ["The most treasured artifact of {s} is {o}.", "{o} is the most treasured artifact of {s}.",
          "{s} treasures {o} above all its other artifacts.", "{s} has {o} as its most treasured artifact.",
          "Of all its artifacts, {s} treasures {o} most."],
         ["the most treasured artifact of {x}", "the artifact that is the most treasured artifact of {x}"],
         ["whose most treasured artifact is {y}", "that has as its most treasured artifact {y}"]),
    _rel("festival", "City", "Event", ["signature festival"],
         ["The signature festival of {s} is {o}.", "{o} is the signature festival of {s}.",
          "{s} has {o} as its signature festival.", "{s} celebrates {o} as its signature festival.",
          "{o} is celebrated as the signature festival of {s}."],
         ["the signature festival of {x}", "the event that is the signature festival of {x}"],
         ["whose signature festival is {y}", "that celebrates as its signature festival {y}"]),
    # ---- Organization ->
    _rel("headquartered_in", "Organization", "City", ["headquarters", "headquartered"],
         ["{s} is headquartered in {o}.", "{o} hosts the headquarters of {s}.", "The headquarters of {s} are in {o}.",
          "{s} has its headquarters in {o}.", "The city of {o} hosts the headquarters of {s}."],
         ["the headquarters city of {x}", "the city that hosts the headquarters of {x}"],
         ["whose headquarters city is {y}", "that is headquartered in {y}"]),
    _rel("founded_by", "Organization", "Person", ["founder", "founded"],
         ["{s} was founded by {o}.", "{o} founded {s}.", "The founder of {s} is {o}.", "{o} is the founder of {s}.",
          "{o} is credited as the founder of {s}."],
         ["the founder of {x}", "the person who founded {x}"],
         ["whose founder is {y}", "that was founded by {y}"]),
    _rel("led_by", "Organization", "Person", ["leader", "leads", "led"],
         ["{s} is led by {o}.", "{o} leads {s}.", "The leader of {s} is {o}.", "{o} is the leader of {s}.",
          "{o} serves as the leader of {s}."],
         ["the leader of {x}", "the person who leads {x}"],
         ["whose leader is {y}", "that is led by {y}"]),
    _rel("parent_org", "Organization", "Organization", ["parent organization", "subsidiary"],
         ["{s} is a subsidiary of {o}.", "{o} is the parent organization of {s}.",
          "The parent organization of {s} is {o}.", "{s} operates as a subsidiary of {o}.",
          "{o} has {s} as a subsidiary."],
         ["the parent organization of {x}", "the organization that is the parent organization of {x}"],
         ["whose parent organization is {y}", "that is a subsidiary of {y}"]),
    _rel("flagship", "Organization", "Artifact", ["flagship product"],
         ["The flagship product of {s} is {o}.", "{o} is the flagship product of {s}.",
          "{s} has {o} as its flagship product.", "{s} counts {o} as its flagship product.",
          "Among the products of {s}, the flagship product is {o}."],
         ["the flagship product of {x}", "the artifact that is the flagship product of {x}"],
         ["whose flagship product is {y}", "that has as its flagship product {y}"]),
    # ---- Artifact ->
    _rel("created_by", "Artifact", "Person", ["creator", "created"],
         ["{s} was created by {o}.", "{o} created {s}.", "The creator of {s} is {o}.", "{o} is the creator of {s}.",
          "{o} is credited as the creator of {s}."],
         ["the creator of {x}", "the person who created {x}"],
         ["whose creator is {y}", "that was created by {y}"]),
    _rel("restored_by", "Artifact", "Person", ["restorer", "restored", "restoration"],
         ["{s} was restored by {o}.", "{o} restored {s}.", "The restorer of {s} is {o}.",
          "{o} is the restorer of {s}.", "{s} owes its restoration to {o}."],
         ["the restorer of {x}", "the person who restored {x}"],
         ["whose restorer is {y}", "that was restored by {y}"]),
    _rel("owned_by", "Artifact", "Organization", ["owner", "owns", "owned"],
         ["{s} is owned by {o}.", "{o} owns {s}.", "The owner of {s} is {o}.", "{o} is the owner of {s}.",
          "{o} is the current owner of {s}."],
         ["the owner of {x}", "the organization that owns {x}"],
         ["whose owner is {y}", "that is owned by {y}"]),
    _rel("unveiled_at", "Artifact", "Event", ["unveiled", "unveiling"],
         ["{s} was unveiled at {o}.", "{o} saw the unveiling of {s}.", "The unveiling of {s} took place at {o}.",
          "{s} was unveiled to the public at {o}.", "{o} is the unveiling event of {s}."],
         ["the unveiling event of {x}", "the event that saw the unveiling of {x}"],
         ["whose unveiling event is {y}", "that was unveiled at {y}"]),
    # ---- Event ->
    _rel("held_in", "Event", "City", ["held", "venue"],
         ["{s} is held in {o}.", "The venue city of {s} is {o}.", "{o} is the venue city of {s}.",
          "{s} is held in the city of {o}.", "{o} serves as the venue city of {s}."],
         ["the venue city of {x}", "the city that is the venue of {x}"],
         ["whose venue city is {y}", "that is held in {y}"]),
    _rel("organized_by", "Event", "Organization", ["organizer", "organizes", "organized"],
         ["{s} is organized by {o}.", "{o} organizes {s}.", "The organizer of {s} is {o}.",
          "{o} is the organizer of {s}.", "{o} serves as the organizer of {s}."],
         ["the organizer of {x}", "the organization that organizes {x}"],
         ["whose organizer is {y}", "that is organized by {y}"]),
    _rel("sponsored_by", "Event", "Organization", ["sponsor", "sponsors", "sponsored"],
         ["{s} is sponsored by {o}.", "{o} sponsors {s}.", "The sponsor of {s} is {o}.", "{o} is the sponsor of {s}.",
          "{o} serves as the sponsor of {s}."],
         ["the sponsor of {x}", "the organization that sponsors {x}"],
         ["whose sponsor is {y}", "that is sponsored by {y}"]),
    _rel("headlined_by", "Event", "Person", ["headliner", "headlines", "headlined"],
         ["{s} is headlined by {o}.", "{o} headlines {s}.", "The headliner of {s} is {o}.",
          "{o} is the headliner of {s}.", "{s} features {o} as its headliner."],
         ["the headliner of {x}", "the person who headlines {x}"],
         ["whose headliner is {y}", "that is headlined by {y}"]),
]}

ATTRIBUTES: dict[str, Attribute] = {a.name: a for a in [
    Attribute("birth_year", "Person", "year", 1850, 2000, ("birth year",),
              ("The birth year of {s} is {v}.", "{s} has a birth year of {v}.", "{v} is the birth year of {s}.",
               "{s} has a recorded birth year of {v}.", "The recorded birth year of {s} is {v}."),
              {"max": ("Which person has the later birth year: {a}, or {b}?",
                       "Of {a} and {b}, which person has the later birth year?"),
               "min": ("Which person has the earlier birth year: {a}, or {b}?",
                       "Of {a} and {b}, which person has the earlier birth year?")}),
    Attribute("population", "City", "count", 20_000, 5_000_000, ("population",),
              ("{s} has a population of {v}.", "The population of {s} is {v}.", "{v} is the population of {s}.",
               "The city of {s} has a population of {v}.", "The recorded population of {s} is {v}."),
              {"max": ("Which city has the larger population: {a}, or {b}?",
                       "Of {a} and {b}, which city has the larger population?"),
               "min": ("Which city has the smaller population: {a}, or {b}?",
                       "Of {a} and {b}, which city has the smaller population?")}),
    Attribute("established_year", "Organization", "year", 1700, 2020, ("established", "year of establishment"),
              ("{s} was established in {v}.", "The year of establishment of {s} is {v}.",
               "{v} is the year of establishment of {s}.", "In {v}, {s} was established.",
               "{s} has a year of establishment of {v}."),
              {"max": ("Which organization was established later: {a}, or {b}?",
                       "Of {a} and {b}, which organization has the later year of establishment?"),
               "min": ("Which organization was established earlier: {a}, or {b}?",
                       "Of {a} and {b}, which organization has the earlier year of establishment?")}),
    Attribute("completed_year", "Artifact", "year", 1400, 2020, ("completed", "year of completion"),
              ("{s} was completed in {v}.", "The year of completion of {s} is {v}.",
               "{v} is the year of completion of {s}.", "In {v}, {s} was completed.",
               "{s} has a year of completion of {v}."),
              {"max": ("Which artifact was completed later: {a}, or {b}?",
                       "Of {a} and {b}, which artifact has the later year of completion?"),
               "min": ("Which artifact was completed earlier: {a}, or {b}?",
                       "Of {a} and {b}, which artifact has the earlier year of completion?")}),
    Attribute("attendance", "Event", "count", 500, 400_000, ("attendance",),
              ("{s} has an attendance of {v}.", "The attendance of {s} is {v}.", "{v} is the attendance of {s}.",
               "{s} draws an attendance of {v}.", "The recorded attendance of {s} is {v}."),
              {"max": ("Which event has the larger attendance: {a}, or {b}?",
                       "Of {a} and {b}, which event has the larger attendance?"),
               "min": ("Which event has the smaller attendance: {a}, or {b}?",
                       "Of {a} and {b}, which event has the smaller attendance?")}),
]}

ATTR_OF_TYPE = {a.subj: a.name for a in ATTRIBUTES.values()}
OUT: dict[str, list[str]] = {t: sorted(r.name for r in RELATIONS.values() if r.subj == t) for t in TYPES}

NAME_RE = r"[A-Z][a-z]+(?: [A-Z][a-z]+)*"
VALUE_RE = r"\d[\d,]*"


def fmt_value(attr: str, v: int) -> str:
    return f"{v:,}" if ATTRIBUTES[attr].kind == "count" else str(v)


def parse_value(s: str) -> int:
    return int(s.replace(",", ""))


def _compile(template: str) -> re.Pattern:
    parts = re.split(r"(\{[sov]\})", template)
    out = []
    for p in parts:
        if p == "{s}":
            out.append(f"(?P<s>{NAME_RE})")
        elif p == "{o}":
            out.append(f"(?P<o>{NAME_RE})")
        elif p == "{v}":
            out.append(f"(?P<v>{VALUE_RE})")
        else:
            out.append(re.escape(p))
    return re.compile("^" + "".join(out) + "$")


@lru_cache(maxsize=1)
def fact_regexes() -> list[tuple[str, int, re.Pattern]]:
    """(relation or attribute name, template index, compiled regex) for every fact template."""
    out = []
    for rel in list(RELATIONS.values()) + list(ATTRIBUTES.values()):
        for i, t in enumerate(rel.facts):
            out.append((rel.name, i, _compile(t)))
    return out


def fact_templates(rel: str) -> tuple[str, ...]:
    return (RELATIONS[rel] if rel in RELATIONS else ATTRIBUTES[rel]).facts


def subj_type(rel: str) -> str:
    return (RELATIONS[rel] if rel in RELATIONS else ATTRIBUTES[rel]).subj
