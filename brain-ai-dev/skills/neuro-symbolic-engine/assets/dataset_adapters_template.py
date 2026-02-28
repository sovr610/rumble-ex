"""
Dataset adapters for converting ProofWriter and FOLIO logic datasets into
the neuro-symbolic engine's internal format.

Adapters ingest dataset splits, parse raw representations (English or FOL
annotations) into SymbolicExample records, and collate them into batched
tensors for the rule engine and grounding pipeline.

Dependencies: torch, json, os, re, pathlib (no external NLP libraries).
"""

from __future__ import annotations

import json
import os
import random
import re
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SymbolicExample:
    """Single example from a logic dataset."""
    entities: List[str] = field(default_factory=list)
    predicates: Dict[str, int] = field(default_factory=dict)  # name -> arity
    rules: List[Dict] = field(default_factory=list)
    query: Optional[Dict] = None
    label: Optional[float] = None  # 1.0=entailed, 0.0=contradicted, 0.5=unknown
    proof_chain: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def num_rules(self) -> int:
        return len(self.rules)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SymbolicExample":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SymbolicBatch:
    """Collated batch of symbolic examples."""
    entity_ids: Tensor       # (B, max_N)
    entity_mask: Tensor      # (B, max_N) boolean
    rules: List[List[Dict]]  # B lists of rule dicts
    queries: List[Optional[Dict]]
    labels: Optional[Tensor] # (B,) or None
    num_entities: List[int]
    metadata: List[Dict]


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class SymbolicDatasetAdapter:
    """Base class for logic dataset adapters."""

    def load(self, split: str = "train",
             max_examples: Optional[int] = None) -> List[SymbolicExample]:
        raise NotImplementedError

    def collate(self, examples: List[SymbolicExample],
                max_entities: int = 32) -> SymbolicBatch:
        """Collate examples into a padded batch."""
        if not examples:
            return SymbolicBatch(
                entity_ids=torch.zeros(0, 0, dtype=torch.long),
                entity_mask=torch.zeros(0, 0, dtype=torch.bool),
                rules=[], queries=[], labels=None, num_entities=[], metadata=[],
            )
        B = len(examples)
        truncated = [ex.entities[:max_entities] for ex in examples]
        max_n = max(len(e) for e in truncated) or 1

        vocab: Dict[str, int] = {}
        idx = 0
        for ents in truncated:
            for name in ents:
                if name not in vocab:
                    vocab[name] = idx
                    idx += 1

        entity_ids = torch.zeros(B, max_n, dtype=torch.long)
        entity_mask = torch.zeros(B, max_n, dtype=torch.bool)
        for i, ents in enumerate(truncated):
            for j, name in enumerate(ents):
                entity_ids[i, j] = vocab[name]
                entity_mask[i, j] = True

        labs = [ex.label for ex in examples]
        labels = torch.tensor(labs, dtype=torch.float32) if all(l is not None for l in labs) else None

        return SymbolicBatch(
            entity_ids=entity_ids, entity_mask=entity_mask,
            rules=[ex.rules for ex in examples],
            queries=[ex.query for ex in examples],
            labels=labels,
            num_entities=[len(e) for e in truncated],
            metadata=[ex.metadata for ex in examples],
        )

    def get_predicate_vocabulary(self) -> Dict[str, int]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# FOL tokenizer
# ---------------------------------------------------------------------------

class FOLTokenizer:
    """Tokenizer for first-order logic strings."""
    QUANTIFIERS: Set[str] = {"\u2200", "forall", "FORALL", "\u2203", "exists", "EXISTS"}
    CONNECTIVES: Set[str] = {
        "\u2227", "and", "AND", "&", "^", "\u2228", "or", "OR", "|",
        "\u2192", "implies", "IMPLIES", "->", "\u2194", "iff", "IFF", "<->",
    }
    NEGATION: Set[str] = {"\u00AC", "not", "NOT", "~"}
    PARENS: Set[str] = {"(", ")"}
    COMMA: Set[str] = {","}
    _SPECIALS: Set[str] = {"(", ")", ",", "\u2200", "\u2203", "\u2227", "\u2228",
                           "\u2192", "\u2194", "\u00AC", "&", "^", "|", "~"}
    _MULTI_OPS: List[str] = ["<->", "->"]

    def tokenize(self, text: str) -> List[str]:
        for op in self._MULTI_OPS:
            text = text.replace(op, f" {op} ")
        result: List[str] = []
        buf: List[str] = []

        def _flush() -> None:
            nonlocal buf
            w = "".join(buf).strip()
            if w:
                result.append(w)
            buf = []

        for ch in text:
            if ch in self._SPECIALS:
                _flush(); result.append(ch)
            elif ch.isspace():
                _flush()
            else:
                buf.append(ch)
        _flush()

        # Split keyword-variable concatenations like "forallx" -> ["forall","x"].
        keywords = sorted(
            list(self.QUANTIFIERS) + list(self.CONNECTIVES) + list(self.NEGATION),
            key=lambda k: -len(k),
        )
        normalised: List[str] = []
        for token in result:
            if token in self._SPECIALS or token in self._MULTI_OPS:
                normalised.append(token); continue
            matched = False
            for kw in keywords:
                if (token.lower().startswith(kw.lower()) and len(token) > len(kw)
                        and kw.isalpha()):
                    rest = token[len(kw):]
                    if rest[0].isalpha() or rest[0] == "_":
                        normalised.extend([kw, rest]); matched = True; break
            if not matched:
                normalised.append(token)
        return normalised

    def normalise_quantifier(self, token: str) -> str:
        upper = token.upper() if token.isascii() else token
        if upper in {"FORALL", "\u2200"}: return "forall"
        if upper in {"EXISTS", "\u2203"}: return "exists"
        return token

    def normalise_connective(self, token: str) -> str:
        m: Dict[str, str] = {
            "\u2227": "and", "AND": "and", "&": "and", "^": "and", "and": "and",
            "\u2228": "or", "OR": "or", "|": "or", "or": "or",
            "\u2192": "implies", "IMPLIES": "implies", "->": "implies", "implies": "implies",
            "\u2194": "iff", "IFF": "iff", "<->": "iff", "iff": "iff",
        }
        return m.get(token, m.get(token.upper(), token))


# ---------------------------------------------------------------------------
# FOL recursive-descent parser
# ---------------------------------------------------------------------------

class FOLParser:
    """Recursive-descent parser for FOL formulas.

    Precedence (loosest to tightest): iff < implies < or < and < not < atom.
    """
    def __init__(self) -> None:
        self._tok = FOLTokenizer()
        self._tokens: List[str] = []
        self._pos: int = 0

    def parse(self, tokens: List[str]) -> Dict:
        self._tokens, self._pos = tokens, 0
        ast = self._parse_formula()
        if self._pos < len(self._tokens):
            raise ValueError(f"Trailing tokens: {self._tokens[self._pos:]}")
        return ast

    def parse_string(self, fol_string: str) -> Dict:
        return self.parse(self._tok.tokenize(fol_string))

    def _peek(self) -> Optional[str]:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _advance(self) -> str:
        t = self._tokens[self._pos]; self._pos += 1; return t

    def _expect(self, expected: str) -> str:
        tok = self._peek()
        if tok != expected:
            raise ValueError(f"Expected '{expected}' at pos {self._pos}, got '{tok}'")
        return self._advance()

    def _parse_formula(self) -> Dict:
        return self._parse_iff()

    def _parse_iff(self) -> Dict:
        left = self._parse_implies()
        while self._peek() and self._tok.normalise_connective(self._peek()) == "iff":
            self._advance(); left = {"type": "iff", "lhs": left, "rhs": self._parse_implies()}
        return left

    def _parse_implies(self) -> Dict:
        left = self._parse_or()
        if self._peek() and self._tok.normalise_connective(self._peek()) == "implies":
            self._advance(); left = {"type": "implies", "lhs": left, "rhs": self._parse_implies()}
        return left

    def _parse_or(self) -> Dict:
        left = self._parse_and()
        while self._peek() and self._tok.normalise_connective(self._peek()) == "or":
            self._advance(); left = {"type": "or", "args": [left, self._parse_and()]}
        return left

    def _parse_and(self) -> Dict:
        left = self._parse_unary()
        while self._peek() and self._tok.normalise_connective(self._peek()) == "and":
            self._advance(); left = {"type": "and", "args": [left, self._parse_unary()]}
        return left

    def _parse_unary(self) -> Dict:
        tok = self._peek()
        if tok and tok in FOLTokenizer.NEGATION:
            self._advance(); return {"type": "not", "body": self._parse_unary()}
        return self._parse_quantified()

    def _parse_quantified(self) -> Dict:
        tok = self._peek()
        if tok and tok in FOLTokenizer.QUANTIFIERS:
            q = self._tok.normalise_quantifier(self._advance())
            var = self._advance()
            if self._peek() in {":", "."}: self._advance()
            return {"type": q, "var": var, "body": self._parse_formula()}
        return self._parse_atom()

    def _parse_atom(self) -> Dict:
        tok = self._peek()
        if tok == "(":
            self._advance(); inner = self._parse_formula(); self._expect(")"); return inner
        if tok and (tok[0].isalpha() or tok.startswith("_")):
            name = self._advance()
            if self._peek() == "(":
                self._advance()
                args: List[str] = []
                if self._peek() != ")":
                    args.append(self._advance())
                    while self._peek() == ",":
                        self._advance(); args.append(self._advance())
                self._expect(")")
                return {"type": "literal", "predicate": name, "vars": args}
            return {"type": "literal", "predicate": name, "vars": []}
        raise ValueError(f"Unexpected token '{tok}' at pos {self._pos}")


# ---------------------------------------------------------------------------
# ProofWriter adapter
# ---------------------------------------------------------------------------

_FACT_PAT = re.compile(r"^(?:The\s+)?(\w+)\s+is\s+(.+?)\.$", re.I)
_BIN_FACT_PAT = re.compile(r"^(?:The\s+)?(\w+)\s+(\w+(?:e?s)?)\s+(?:the\s+)?(\w+)\.$", re.I)
_RULE_SIMPLE = re.compile(
    r"^If\s+(?:something|someone|an?\s+\w+)\s+is\s+(.+?)\s+"
    r"(?:then|,)\s+(?:it|they)\s+(?:is|are)\s+(.+?)\.$", re.I)
_RULE_COMBINED = re.compile(
    r"^If\s+(?:something|someone)\s+is\s+(.+?)\s+and\s+"
    r"(?:(?:it|they)\s+(?:is|are)\s+)?(.+?)\s+"
    r"(?:then|,)\s+(?:it|they)\s+(?:is|are)\s+(.+?)\.$", re.I)
_RULE_BINARY = re.compile(
    r"^If\s+(?:something|someone)\s+(\w+(?:e?s)?)\s+(?:the\s+)?(\w+)\s+"
    r"(?:then|,)\s+(?:it|they)\s+(?:is|are)\s+(.+?)\.$", re.I)


class ProofWriterAdapter(SymbolicDatasetAdapter):
    """Adapter for ProofWriter dataset.

    Parses English facts/rules into constrained logic form.
    Supports structured variants with pre-parsed logic.
    """
    def __init__(self, data_dir: str, use_structured: bool = True,
                 max_depth: Optional[int] = None) -> None:
        self.data_dir = data_dir
        self.use_structured = use_structured
        self.max_depth = max_depth
        self._predicate_vocab: Dict[str, int] = {}

    def load(self, split: str = "train",
             max_examples: Optional[int] = None) -> List[SymbolicExample]:
        path = self._find_file(split)
        raws = self._read_jsonl(path)
        if self.max_depth is not None:
            raws = [r for r in raws if r.get("meta", {}).get("depth", 0) <= self.max_depth]
        if max_examples:
            raws = raws[:max_examples]
        results: List[SymbolicExample] = []
        for i, raw in enumerate(raws):
            ex = self._parse_example(raw, i)
            if ex:
                results.append(ex)
                self._predicate_vocab.update(ex.predicates)
        return results

    def get_predicate_vocabulary(self) -> Dict[str, int]:
        return dict(self._predicate_vocab)

    def _find_file(self, split: str) -> str:
        for name in [f"{split}.jsonl", f"{split}.json",
                     f"depth-5/{split}.jsonl", f"depth-5/{split}.json"]:
            p = os.path.join(self.data_dir, name)
            if os.path.isfile(p): return p
        raise FileNotFoundError(f"No file for split '{split}' in {self.data_dir}")

    @staticmethod
    def _read_jsonl(path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline().strip(); f.seek(0)
            if first.startswith("["): return json.load(f)
            return [json.loads(ln) for ln in f if ln.strip()]

    def _parse_example(self, raw: Dict, eid: int) -> Optional[SymbolicExample]:
        if self.use_structured and "triples" in raw:
            return self._parse_structured(raw, eid)
        return self._parse_english(raw, eid)

    def _parse_structured(self, raw: Dict, eid: int) -> Optional[SymbolicExample]:
        entities: Set[str] = set()
        predicates: Dict[str, int] = {}
        facts: List[Dict] = []
        for _, triple in raw.get("triples", {}).items():
            text = triple if isinstance(triple, str) else triple.get("text", "")
            lit = self._parse_fact(text)
            if lit:
                facts.append(lit)
                for v in lit.get("vars", []): entities.add(v)
                predicates[lit["predicate"]] = len(lit["vars"])
        rules: List[Dict] = []
        for _, robj in raw.get("rules", {}).items():
            text = robj if isinstance(robj, str) else robj.get("text", "")
            r = self._parse_rule(text)
            if r: rules.append(r)
        all_rules = facts + rules
        query, label = None, None
        qs = raw.get("questions", {})
        if qs:
            qobj = qs[next(iter(qs))]
            qt = qobj if isinstance(qobj, str) else qobj.get("text", "")
            ql = "" if isinstance(qobj, str) else qobj.get("label", "")
            query = self._parse_fact(qt)
            label = self._label_to_float(ql)
        return SymbolicExample(
            entities=sorted(entities), predicates=predicates, rules=all_rules,
            query=query, label=label, metadata={"source": "proofwriter", "example_id": eid,
            "depth": raw.get("meta", {}).get("depth", 0), "format": "structured"})

    def _parse_english(self, raw: Dict, eid: int) -> Optional[SymbolicExample]:
        theory = raw.get("theory", raw.get("context", ""))
        sents = [s.strip() + "." for s in theory.split(".") if s.strip()]
        fact_sents = [s for s in sents if not s.lower().startswith("if ")]
        rule_sents = [s for s in sents if s.lower().startswith("if ")]
        entities = self._extract_entities(fact_sents)
        predicates: Dict[str, int] = {}
        facts: List[Dict] = []
        for s in fact_sents:
            lit = self._parse_fact(s)
            if lit:
                facts.append(lit)
                predicates[lit["predicate"]] = len(lit["vars"])
        rules: List[Dict] = []
        for s in rule_sents:
            r = self._parse_rule(s)
            if r: rules.append(r)
        query, label = None, None
        qs = raw.get("questions", {})
        if qs:
            qobj = (qs[next(iter(qs))] if isinstance(qs, dict)
                    else qs[0] if isinstance(qs, list) and qs else None)
            if qobj:
                qt = qobj if isinstance(qobj, str) else qobj.get("text", "")
                ql = "" if isinstance(qobj, str) else qobj.get("label", "")
                query = self._parse_fact(qt); label = self._label_to_float(ql)
        return SymbolicExample(
            entities=entities, predicates=predicates, rules=facts + rules,
            query=query, label=label, metadata={"source": "proofwriter",
            "example_id": eid, "depth": raw.get("meta", {}).get("depth", 0),
            "format": "english"})

    def _parse_fact(self, sentence: str) -> Optional[Dict]:
        """Parse fact: 'The cat is blue.' -> literal dict."""
        s = sentence.strip().rstrip("?")
        if not s.endswith("."): s += "."
        m = _FACT_PAT.match(s)
        if m:
            return {"type": "literal", "predicate": self._normalize_predicate("is_" + m.group(2).strip().lower()),
                    "vars": [m.group(1).lower()]}
        m = _BIN_FACT_PAT.match(s)
        if m:
            return {"type": "literal", "predicate": self._normalize_predicate(m.group(2).lower()),
                    "vars": [m.group(1).lower(), m.group(3).lower()]}
        return None

    def _parse_rule(self, sentence: str) -> Optional[Dict]:
        """Parse rule into forall/implies AST dict."""
        s = sentence.strip()
        m = _RULE_COMBINED.match(s)
        if m:
            a1 = self._normalize_predicate("is_" + m.group(1).strip().lower())
            a2 = self._normalize_predicate("is_" + m.group(2).strip().lower())
            c = self._normalize_predicate("is_" + m.group(3).strip().lower())
            return {"type": "forall", "var": "x", "body": {"type": "implies",
                "lhs": {"type": "and", "args": [
                    {"type": "literal", "predicate": a1, "vars": ["x"]},
                    {"type": "literal", "predicate": a2, "vars": ["x"]}]},
                "rhs": {"type": "literal", "predicate": c, "vars": ["x"]}}}
        m = _RULE_SIMPLE.match(s)
        if m:
            a = self._normalize_predicate("is_" + m.group(1).strip().lower())
            c = self._normalize_predicate("is_" + m.group(2).strip().lower())
            return {"type": "forall", "var": "x", "body": {"type": "implies",
                "lhs": {"type": "literal", "predicate": a, "vars": ["x"]},
                "rhs": {"type": "literal", "predicate": c, "vars": ["x"]}}}
        m = _RULE_BINARY.match(s)
        if m:
            verb = self._normalize_predicate(m.group(1).lower())
            obj = m.group(2).lower()
            c = self._normalize_predicate("is_" + m.group(3).strip().lower())
            return {"type": "forall", "var": "x", "body": {"type": "implies",
                "lhs": {"type": "literal", "predicate": verb, "vars": ["x", obj]},
                "rhs": {"type": "literal", "predicate": c, "vars": ["x"]}}}
        return None

    def _extract_entities(self, facts: List[str]) -> List[str]:
        ents: Set[str] = set()
        for s in facts:
            words = s.strip().rstrip(".").split()
            idx = 0
            while idx < len(words) and words[idx].lower() in {"the", "a", "an"}: idx += 1
            if idx < len(words):
                e = words[idx].lower()
                if e not in {"something", "someone", "if", "then"}: ents.add(e)
            m = _BIN_FACT_PAT.match(s if s.endswith(".") else s + ".")
            if m: ents.add(m.group(1).lower()); ents.add(m.group(3).lower())
        return sorted(ents)

    def _normalize_predicate(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"[\s\-]+", "_", text)
        return re.sub(r"[^a-z0-9_]", "", text)

    @staticmethod
    def _label_to_float(s: str) -> Optional[float]:
        if not s: return None
        s = s.strip().lower()
        if s in {"true", "entailed", "1", "1.0"}: return 1.0
        if s in {"false", "contradicted", "0", "0.0"}: return 0.0
        if s in {"unknown", "neutral", "0.5"}: return 0.5
        return None


# ---------------------------------------------------------------------------
# FOLIO adapter
# ---------------------------------------------------------------------------

class FOLIOAdapter(SymbolicDatasetAdapter):
    """Adapter for FOLIO dataset. Loads FOL annotations and translates to internal AST."""

    def __init__(self, data_dir: str, use_fol: bool = True) -> None:
        self.data_dir = data_dir
        self.use_fol = use_fol
        self._predicate_vocab: Dict[str, int] = {}
        self._parser = FOLParser()

    def load(self, split: str = "train",
             max_examples: Optional[int] = None) -> List[SymbolicExample]:
        path = self._find_file(split)
        raws = self._read_jsonl(path)
        if max_examples: raws = raws[:max_examples]
        results: List[SymbolicExample] = []
        for i, raw in enumerate(raws):
            ex = self._parse_example(raw, i)
            if ex:
                results.append(ex)
                self._predicate_vocab.update(ex.predicates)
        return results

    def get_predicate_vocabulary(self) -> Dict[str, int]:
        return dict(self._predicate_vocab)

    def _find_file(self, split: str) -> str:
        for name in [f"{split}.jsonl", f"{split}.json", f"folio-{split}.jsonl",
                     f"folio-{split}.json", f"folio_{split}.jsonl"]:
            p = os.path.join(self.data_dir, name)
            if os.path.isfile(p): return p
        raise FileNotFoundError(f"No file for split '{split}' in {self.data_dir}")

    @staticmethod
    def _read_jsonl(path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline().strip(); f.seek(0)
            if first.startswith("["): return json.load(f)
            return [json.loads(ln) for ln in f if ln.strip()]

    def _parse_example(self, raw: Dict, eid: int) -> Optional[SymbolicExample]:
        prem_fol = raw.get("premises_fol", raw.get("premises-FOL", []))
        conc_fol = raw.get("conclusion_fol", raw.get("conclusion-FOL", ""))
        label_str = raw.get("label", raw.get("answer", ""))
        if not self.use_fol or not prem_fol:
            return SymbolicExample(entities=[], predicates={}, rules=[], label=self._label_to_float(label_str),
                metadata={"source": "folio", "example_id": eid, "format": "nl_fallback"})
        rules, preds, ents = [], {}, set()
        for fol_str in prem_fol:
            if not fol_str or not fol_str.strip(): continue
            try:
                ast = self._parser.parse_string(fol_str)
                rules.append(ast)
                preds.update(self._collect_predicates(ast))
                ents.update(self._collect_constants(ast))
            except (ValueError, IndexError): continue
        query = None
        if conc_fol and conc_fol.strip():
            try:
                query = self._parser.parse_string(conc_fol)
                preds.update(self._collect_predicates(query))
                ents.update(self._collect_constants(query))
            except (ValueError, IndexError): pass
        return SymbolicExample(
            entities=sorted(ents), predicates=preds, rules=rules, query=query,
            label=self._label_to_float(label_str),
            metadata={"source": "folio", "example_id": eid})

    def _parse_fol(self, fol_string: str) -> Dict:
        return self._parser.parse_string(fol_string)

    def _tokenize_fol(self, fol_string: str) -> List[str]:
        return self._parser._tok.tokenize(fol_string)

    def _extract_predicates(self, fol_strings: List[str]) -> Dict[str, int]:
        preds: Dict[str, int] = {}
        for s in fol_strings:
            if not s or not s.strip(): continue
            try: preds.update(self._collect_predicates(self._parser.parse_string(s)))
            except (ValueError, IndexError): continue
        return preds

    def _map_constants(self, fol_ast: Dict) -> Tuple[Dict, List[str]]:
        return fol_ast, sorted(self._collect_constants(fol_ast))

    @staticmethod
    def _collect_predicates(ast: Dict) -> Dict[str, int]:
        preds: Dict[str, int] = {}
        def walk(n: Dict) -> None:
            t = n.get("type", "")
            if t == "literal":
                p = n.get("predicate", "")
                if p: preds[p] = len(n.get("vars", []))
            elif t in ("forall", "exists", "not"):
                b = n.get("body"); b and walk(b)
            elif t in ("implies", "iff"):
                l, r = n.get("lhs"), n.get("rhs")
                l and walk(l); r and walk(r)
            elif t in ("and", "or"):
                for a in n.get("args", []): walk(a)
        walk(ast); return preds

    @staticmethod
    def _collect_constants(ast: Dict) -> Set[str]:
        constants: Set[str] = set()
        def walk(n: Dict, bv: Set[str]) -> None:
            t = n.get("type", "")
            if t == "literal":
                for v in n.get("vars", []):
                    if v not in bv and (len(v) > 1 or v[0].isupper()): constants.add(v)
            elif t in ("forall", "exists"):
                walk(n.get("body", {}), bv | {n.get("var", "")})
            elif t == "not":
                b = n.get("body"); b and walk(b, bv)
            elif t in ("implies", "iff"):
                l, r = n.get("lhs"), n.get("rhs")
                l and walk(l, bv); r and walk(r, bv)
            elif t in ("and", "or"):
                for a in n.get("args", []): walk(a, bv)
        walk(ast, set()); return constants

    @staticmethod
    def _label_to_float(s: str) -> Optional[float]:
        if not s: return None
        s = s.strip().lower()
        if s in {"true", "entailed", "1", "1.0"}: return 1.0
        if s in {"false", "contradicted", "0", "0.0"}: return 0.0
        if s in {"unknown", "neutral", "uncertain", "0.5"}: return 0.5
        return None


# ---------------------------------------------------------------------------
# Preprocessing cache
# ---------------------------------------------------------------------------

class PreprocessingCache:
    """Cache preprocessed ASTs to disk for fast reload."""

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, re.sub(r"[^a-zA-Z0-9_\-]", "_", key) + ".json")

    def exists(self, key: str) -> bool:
        return os.path.isfile(self._path(key))

    def save(self, key: str, examples: List[SymbolicExample]) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self._path(key), "w", encoding="utf-8") as f:
            json.dump([ex.to_dict() for ex in examples], f, indent=2, default=str)

    def load(self, key: str) -> Optional[List[SymbolicExample]]:
        p = self._path(key)
        if not os.path.isfile(p): return None
        with open(p, "r", encoding="utf-8") as f:
            return [SymbolicExample.from_dict(d) for d in json.load(f)]


# ---------------------------------------------------------------------------
# Synthetic dataset generator
# ---------------------------------------------------------------------------

_PROPS = ["red", "blue", "green", "big", "small", "cold", "hot", "kind", "nice",
          "rough", "smooth", "round", "quiet", "young"]
_RELS = ["chases", "sees", "likes", "visits", "needs", "eats"]
_ENTS = ["cat", "dog", "bear", "lion", "tiger", "rabbit", "mouse", "cow",
         "frog", "squirrel", "eagle", "wolf", "deer", "elephant", "penguin"]


class SyntheticFOLDataset:
    """Generate synthetic FOL examples for testing."""

    def __init__(self, num_entities: int = 10, num_predicates: int = 5,
                 num_relations: int = 3, num_rules: int = 5, seed: int = 42) -> None:
        self.num_entities = min(num_entities, len(_ENTS))
        self.num_predicates = min(num_predicates, len(_PROPS))
        self.num_relations = min(num_relations, len(_RELS))
        self.num_rules = num_rules
        self.seed = seed

    def generate(self, num_examples: int = 100) -> List[SymbolicExample]:
        rng = random.Random(self.seed)
        prop_pool = _PROPS[:self.num_predicates]
        rel_pool = _RELS[:self.num_relations]
        ent_pool = _ENTS[:self.num_entities]
        examples: List[SymbolicExample] = []
        for eid in range(num_examples):
            n_e = rng.randint(max(2, self.num_entities // 2), self.num_entities)
            entities = rng.sample(ent_pool, n_e)
            facts, gt = [], {e: set() for e in entities}
            for e in entities:
                for p in rng.sample(prop_pool, rng.randint(1, max(1, len(prop_pool) // 2))):
                    pn = f"is_{p}"
                    facts.append({"type": "literal", "predicate": pn, "vars": [e]})
                    gt[e].add(pn)
            for _ in range(rng.randint(0, max(1, n_e // 2))):
                e1, e2 = rng.sample(entities, 2)
                facts.append({"type": "literal", "predicate": rng.choice(rel_pool), "vars": [e1, e2]})
            rules: List[Dict] = []
            for _ in range(self.num_rules):
                ap, cp = rng.choice(prop_pool), rng.choice(prop_pool)
                if ap == cp: continue
                apn, cpn = f"is_{ap}", f"is_{cp}"
                rules.append({"type": "forall", "var": "x", "body": {"type": "implies",
                    "lhs": {"type": "literal", "predicate": apn, "vars": ["x"]},
                    "rhs": {"type": "literal", "predicate": cpn, "vars": ["x"]}}})
                for e in entities:
                    if apn in gt[e]: gt[e].add(cpn)
            qe = rng.choice(entities)
            qp = f"is_{rng.choice(prop_pool)}"
            query = {"type": "literal", "predicate": qp, "vars": [qe]}
            label = 1.0 if qp in gt[qe] else 0.0
            preds = {f["predicate"]: len(f["vars"]) for f in facts}
            for p in [f"is_{pr}" for pr in prop_pool]:
                preds.setdefault(p, 1)
            examples.append(SymbolicExample(
                entities=sorted(entities), predicates=preds, rules=facts + rules,
                query=query, label=label,
                metadata={"source": "synthetic", "example_id": eid,
                           "ground_truth": {e: sorted(ps) for e, ps in gt.items()}}))
        return examples


# ---------------------------------------------------------------------------
# Utility: AST pretty-printer
# ---------------------------------------------------------------------------

def ast_to_string(ast: Dict) -> str:
    t = ast.get("type", "?")
    if t == "literal":
        args = ast.get("vars", [])
        p = ast.get("predicate", "?")
        return f"{p}({', '.join(args)})" if args else p
    if t in ("forall", "exists"):
        sym = "\u2200" if t == "forall" else "\u2203"
        return f"{sym}{ast.get('var', '?')}. {ast_to_string(ast['body'])}"
    if t == "not":
        return f"\u00AC{ast_to_string(ast['body'])}"
    if t in ("implies", "iff"):
        op = "\u2192" if t == "implies" else "\u2194"
        return f"({ast_to_string(ast['lhs'])} {op} {ast_to_string(ast['rhs'])})"
    if t in ("and", "or"):
        op = "\u2227" if t == "and" else "\u2228"
        return f"({f' {op} '.join(ast_to_string(a) for a in ast.get('args', []))})"
    return f"<{t}>"


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:
    passed = failed = total = 0
    def check(cond: bool, name: str) -> None:
        nonlocal passed, failed, total; total += 1
        if cond: passed += 1; print(f"  [PASS] {name}")
        else: failed += 1; print(f"  [FAIL] {name}")

    print("=" * 70)
    print("Dataset Adapters Template Self-Tests")
    print("=" * 70)

    # 1. SymbolicExample creation
    print("\n--- Test 1: SymbolicExample creation ---")
    ex = SymbolicExample(entities=["cat", "dog"], predicates={"is_blue": 1, "chases": 2},
        rules=[{"type": "literal", "predicate": "is_blue", "vars": ["cat"]},
               {"type": "forall", "var": "x", "body": {"type": "implies",
                "lhs": {"type": "literal", "predicate": "is_blue", "vars": ["x"]},
                "rhs": {"type": "literal", "predicate": "is_cold", "vars": ["x"]}}}],
        query={"type": "literal", "predicate": "is_cold", "vars": ["cat"]},
        label=1.0, proof_chain=[0, 1], metadata={"source": "test", "depth": 1})
    check(ex.num_entities == 2, f"num_entities==2 (got {ex.num_entities})")
    check(ex.num_rules == 2, f"num_rules==2 (got {ex.num_rules})")
    d = ex.to_dict(); rt = SymbolicExample.from_dict(d)
    check(rt.entities == ex.entities and rt.label == 1.0, "round-trip preserves fields")

    # 2. Collation with padding
    print("\n--- Test 2: SymbolicBatch collation ---")
    ex1 = SymbolicExample(entities=["cat", "dog", "bird"], predicates={"is_blue": 1},
        rules=[{"type": "literal", "predicate": "is_blue", "vars": ["cat"]}],
        query={"type": "literal", "predicate": "is_blue", "vars": ["cat"]}, label=1.0, metadata={})
    ex2 = SymbolicExample(entities=["fish"], predicates={"is_red": 1},
        rules=[{"type": "literal", "predicate": "is_red", "vars": ["fish"]}], label=0.0, metadata={})
    adapter = SymbolicDatasetAdapter()
    batch = adapter.collate([ex1, ex2])
    check(batch.entity_ids.shape == (2, 3), f"ids shape (2,3) got {batch.entity_ids.shape}")
    check(batch.entity_mask[0].sum().item() == 3, "ex1 has 3 valid")
    check(batch.entity_mask[1].sum().item() == 1, "ex2 has 1 valid")
    check(batch.labels is not None and batch.labels.shape == (2,), "labels shape (2,)")
    check(batch.num_entities == [3, 1], "num_entities correct")

    # 3. ProofWriter fact parsing
    print("\n--- Test 3: ProofWriter fact parsing ---")
    pw = ProofWriterAdapter(data_dir="/tmp/nonexistent")
    fact = pw._parse_fact("The cat is blue.")
    check(fact is not None and fact["predicate"] == "is_blue" and fact["vars"] == ["cat"],
          "parsed 'The cat is blue.' correctly")
    fb = pw._parse_fact("The cat chases the dog.")
    check(fb is not None and fb["predicate"] == "chases" and fb["vars"] == ["cat", "dog"],
          "parsed binary fact correctly")

    # 4. ProofWriter rule parsing
    print("\n--- Test 4: ProofWriter rule parsing ---")
    rule = pw._parse_rule("If something is blue then it is cold.")
    check(rule is not None and rule["type"] == "forall", "rule is forall")
    check(rule["body"]["type"] == "implies", "body is implies")
    check(rule["body"]["lhs"]["predicate"] == "is_blue", "lhs pred == is_blue")
    check(rule["body"]["rhs"]["predicate"] == "is_cold", "rhs pred == is_cold")
    rc = pw._parse_rule("If something is blue and it is big then it is nice.")
    check(rc is not None and rc["body"]["lhs"]["type"] == "and", "combined rule has AND")

    # 5. Entity extraction
    print("\n--- Test 5: ProofWriter entity extraction ---")
    ents = pw._extract_entities(["The cat is blue.", "The dog is red.", "The cat chases the dog."])
    check("cat" in ents and "dog" in ents and len(ents) >= 2, f"extracted entities: {ents}")

    # 6. Predicate normalization
    print("\n--- Test 6: Predicate normalization ---")
    check(pw._normalize_predicate("is blue") == "is_blue", "'is blue' -> 'is_blue'")
    check(pw._normalize_predicate("Is Big") == "is_big", "'Is Big' -> 'is_big'")
    check(pw._normalize_predicate("  nice  ") == "nice", "'  nice  ' -> 'nice'")

    # 7. FOL tokenization
    print("\n--- Test 7: FOL tokenization ---")
    tok = FOLTokenizer()
    tokens = tok.tokenize("\u2200x (Bird(x) \u2192 CanFly(x))")
    check("\u2200" in tokens and "Bird" in tokens and "\u2192" in tokens and "CanFly" in tokens,
          f"correct tokens: {tokens}")
    ta = tok.tokenize("forall x (P(x) implies Q(x))")
    check("forall" in ta and "implies" in ta, f"ASCII tokens: {ta}")

    # 8. FOL parsing
    print("\n--- Test 8: FOL parsing ---")
    parser = FOLParser()
    ast = parser.parse_string("\u2200x (Bird(x) \u2192 CanFly(x))")
    check(ast["type"] == "forall" and ast["var"] == "x", "forall x")
    check(ast["body"]["type"] == "implies", "body is implies")
    check(ast["body"]["lhs"]["predicate"] == "Bird", "lhs == Bird")
    check(ast["body"]["rhs"]["predicate"] == "CanFly", "rhs == CanFly")
    ast_and = parser.parse_string("P(a) \u2227 Q(b)")
    check(ast_and["type"] == "and" and len(ast_and["args"]) == 2, "conjunction parsed")
    ast_neg = parser.parse_string("\u00ACP(x)")
    check(ast_neg["type"] == "not", "negation parsed")
    ast_ex = parser.parse_string("\u2203x Bird(x)")
    check(ast_ex["type"] == "exists", "existential parsed")

    # 9. FOLIO constant mapping
    print("\n--- Test 9: FOLIO constant mapping ---")
    folio = FOLIOAdapter(data_dir="/tmp/nonexistent")
    _, el = folio._map_constants(parser.parse_string("\u2200x (Bird(x) \u2192 CanFly(x))"))
    check(len(el) == 0, "no constants in fully quantified formula")
    _, el2 = folio._map_constants(parser.parse_string("Likes(Alice, Bob)"))
    check("Alice" in el2 and "Bob" in el2, f"constants: {el2}")

    # 10. FOLIO predicate vocabulary extraction
    print("\n--- Test 10: FOLIO predicate extraction ---")
    pv = folio._extract_predicates(["\u2200x (Bird(x) \u2192 CanFly(x))", "Likes(Alice, Bob)",
                                     "\u2203x (Cat(x) \u2227 Black(x))"])
    check("Bird" in pv and "CanFly" in pv, "Bird, CanFly in vocab")
    check(pv.get("Likes") == 2 and pv.get("Cat") == 1, "arities correct")

    # 11. Synthetic dataset
    print("\n--- Test 11: SyntheticFOLDataset ---")
    synth = SyntheticFOLDataset(num_entities=8, num_predicates=5, num_relations=2, num_rules=4, seed=42)
    se = synth.generate(50)
    check(len(se) == 50, f"generated 50 (got {len(se)})")
    for i, s in enumerate(se[:3]):
        check(s.num_entities >= 2 and s.query is not None and s.label in (0.0, 1.0),
              f"example {i} valid: {s.num_entities} entities, label={s.label}")

    # 12. Variable-length collation
    print("\n--- Test 12: Variable-length entity collation ---")
    exs = SymbolicExample(entities=["a"], predicates={"P": 1},
        rules=[{"type": "literal", "predicate": "P", "vars": ["a"]}], label=1.0, metadata={})
    exl = SymbolicExample(entities=["a", "b", "c", "d", "e"], predicates={"P": 1},
        rules=[{"type": "literal", "predicate": "P", "vars": ["a"]}], label=0.0, metadata={})
    bv = adapter.collate([exs, exl])
    check(bv.entity_ids.shape[1] == 5, f"max_N==5 got {bv.entity_ids.shape[1]}")
    check(bv.entity_mask[0].sum().item() == 1 and bv.entity_mask[1].sum().item() == 5, "masks correct")
    bt = adapter.collate([exl], max_entities=3)
    check(bt.entity_ids.shape[1] == 3 and bt.entity_mask[0].sum().item() == 3, "truncation works")
    be = adapter.collate([])
    check(be.entity_ids.shape[0] == 0, "empty batch")

    # 13. Preprocessing cache round-trip
    print("\n--- Test 13: Preprocessing cache round-trip ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = PreprocessingCache(tmpdir)
        check(not cache.exists("k"), "key missing initially")
        cache.save("k", [ex1, ex2])
        check(cache.exists("k"), "key exists after save")
        loaded = cache.load("k")
        check(loaded is not None and len(loaded) == 2, f"loaded 2 examples")
        check(loaded[0].entities == ex1.entities, "entities preserved")
        check(cache.load("missing") is None, "missing key returns None")

    # Summary
    print("\n" + "=" * 70)
    print(f"{passed}/{total} self-tests passed")
    if failed > 0:
        print(f"{failed} test(s) FAILED"); exit(1)
    else:
        print("All tests passed.")
    print("=" * 70)


if __name__ == "__main__":
    _run_self_tests()
