"""T2 (task 10): gloss -> Thai with Typhoon few-shot, following "ข้อตกลงสำหรับ T2" in README.md.

- few-shot examples: train rows only (gloss_words -> thai_refs[--ref-index]), round-robin over grammar groups
- CTC hyps go through t1_rules.normalize() (word() + collapse repeated adjacent words) = the same input T1 gets
- one request per unique gloss string, temperature 0
Outputs (nlp/outputs/):
  t2_gold.jsonl                               {"gloss_seq_id", "t2"}         all 62 sequences
  t2_ctc_template_C1-x16_seed{0,1,2}.jsonl    {"video_id", "t2_from_hyp"}   45 clips per seed
  t2_meta.json                                model, date, base_url, prompt, shots, input rule

    python nlp/run_t2.py --dry-run             # print the prompt, no API call
    python nlp/run_t2.py --only val            # try the prompt on val (6) before the full run
    python nlp/run_t2.py                       # full run (key in TYPHOON_API_KEY)
    python nlp/run_t2.py --local               # local Ollama, no key
"""

import argparse
import datetime
import json
import os
import re
import sys
import urllib.request
from itertools import zip_longest
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from t1_rules import normalize  # noqa: E402  (same input rule as T1)

SENTENCES = BASE / "annotations" / "sentences_for_t2.jsonl"
CTC_DIR = BASE.parent / "results" / "preds" / "cslr"
CTC_FILES = [f"template_C1-x16_seed{s}_test.json" for s in (0, 1, 2)]
OUT = BASE / "outputs"

API_URL = "https://api.opentyphoon.ai/v1"
API_MODEL = "typhoon-v2.5-30b-a3b-instruct"
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/") + "/v1"
OLLAMA_MODEL = "scb10x/typhoon2.5-qwen3-4b"

SYSTEM = (
    "คุณเป็นผู้แปลภาษามือไทยเป็นภาษาไทยเขียน\n"
    "อินพุตคือลำดับ gloss (คำภาษามือเรียงตามไวยากรณ์ภาษามือ ซึ่งมักต่างจากภาษาไทย เช่น กรรมขึ้นก่อน กริยาอยู่ท้าย)\n"
    "ให้เรียงใหม่เป็นประโยคภาษาไทยที่เป็นธรรมชาติ 1 ประโยค\n"
    "กฎ: ใช้เฉพาะความหมายที่มีใน gloss ห้ามเติมข้อมูลใหม่ (เช่น เวลา สถานที่ คนที่ไม่มีใน gloss)\n"
    'ตอบเป็น JSON บรรทัดเดียวเท่านั้น: {"thai": "<ประโยค>"}'
)


def rows():
    return [json.loads(line) for line in SENTENCES.open(encoding="utf-8")]


KEEP_RULE = "กฎเพิ่ม: ต้องถ่ายทอดความหมายของ gloss ให้ครบทุกคำ ห้ามตัดคำใดทิ้ง\n"  # --keep-words (ablation)


def system(keep=False):
    return SYSTEM.replace("ตอบเป็น JSON", KEEP_RULE + "ตอบเป็น JSON") if keep else SYSTEM


def pick_shots(n, ref_index, exclude=None):
    """exclude = grammar_group_id to hold out (tune_t2.py leave-one-group-out)"""
    by_group = {}
    for r in rows():
        if r["split"] == "train" and r["grammar_group_id"] != exclude:  # contract: never val/test
            by_group.setdefault(r["grammar_group_id"], []).append(r)
    rr = [r for tier in zip_longest(*by_group.values()) for r in tier if r]
    return [(" ".join(r["gloss_words"]), r["thai_refs"][ref_index]) for r in rr[:n]]


def messages(gloss, shots, keep=False):
    m = [{"role": "system", "content": system(keep)}]
    for g, t in shots:
        m += [{"role": "user", "content": f"gloss: {g}"},
              {"role": "assistant", "content": json.dumps({"thai": t}, ensure_ascii=False)}]
    return m + [{"role": "user", "content": f"gloss: {gloss}"}]


def parse(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)  # Qwen3-based models may emit reasoning
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0))["thai"].strip() if m else text.strip()
    except (json.JSONDecodeError, KeyError):
        return text.strip()  # keep the raw answer; it shows up in error analysis instead of crashing


def write_jsonl(path, recs):
    with path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("->", path.relative_to(BASE.parent), len(recs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help=f"local Ollama ({OLLAMA_MODEL}), no key")
    ap.add_argument("--model")
    ap.add_argument("--tag", help="output file prefix (default t2, or t2local4b with --local); record keys stay the same")
    ap.add_argument("--shots", type=int, default=20)
    ap.add_argument("--ref-index", type=int, default=0, help="which of thai_refs is the example answer")
    ap.add_argument("--keep-words", action="store_true", help="add the keep-every-gloss-word rule to the prompt")
    ap.add_argument("--only", choices=["val"], help="translate val only and print, write nothing")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    shots = pick_shots(a.shots, a.ref_index)
    if a.dry_run:
        for m in messages("<gloss ที่จะแปล>", shots, a.keep_words):
            print(f"[{m['role']}] {m['content']}")
        print(f"\n{len(shots)} shots")
        return

    base_url, model = (OLLAMA_URL, a.model or OLLAMA_MODEL) if a.local else (API_URL, a.model or API_MODEL)
    tag = a.tag or ("t2local4b" if a.local else "t2")  # keeps a local run from overwriting the API's t2_* files
    key = "ollama" if a.local else os.environ.get("TYPHOON_API_KEY")  # Ollama ignores the key
    if not key:
        sys.exit("set TYPHOON_API_KEY first, or use --local")
    if a.list_models:  # Typhoon returns a bare list, which the OpenAI SDK can't parse
        req = urllib.request.Request(f"{base_url}/models", headers={"Authorization": f"Bearer {key}", "User-Agent": "sign2say-nlp"})
        d = json.load(urllib.request.urlopen(req, timeout=30))
        print("\n".join(sorted(m["id"] for m in (d if isinstance(d, list) else d["data"]))))
        return

    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=base_url, max_retries=5)
    cache = {}

    def t2(words):
        g = " ".join(words)
        if g not in cache:
            # 1024: Qwen3-based models may spend tokens on reasoning before the answer
            r = client.chat.completions.create(model=model, messages=messages(g, shots, a.keep_words), temperature=0, max_tokens=1024)
            ch = r.choices[0]
            if ch.finish_reason == "length":
                print(f"WARNING: answer cut off for {g!r}")
            cache[g] = parse(ch.message.content)
            print(f"  {g}  ->  {cache[g]}")
        return cache[g]

    all_rows = rows()
    if a.only == "val":
        for r in all_rows:
            if r["split"] == "val":
                t2(r["gloss_words"])
                print(f"     refs: {r['thai_refs']}")
        return

    write_jsonl(OUT / f"{tag}_gold.jsonl", [{"gloss_seq_id": r["gloss_seq_id"], "t2": t2(normalize(r["gloss_words"]))}
                                        for r in all_rows])
    for f in CTC_FILES:
        preds = json.loads((CTC_DIR / f).read_text(encoding="utf-8"))
        write_jsonl(OUT / f"{tag}_ctc_{f.removesuffix('_test.json')}.jsonl",
                    [{"video_id": p["video_id"], "t2_from_hyp": t2(normalize(p["hyp"].split()))} for p in preds])

    meta = OUT / f"{tag}_meta.json"
    meta.write_text(json.dumps({
        "tag": tag, "model": model, "base_url": base_url, "local_ollama": a.local, "temperature": 0, "max_tokens": 1024,
        "run_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_rule": "t1_rules.normalize(): word() + collapse repeated adjacent words (same as T1)",
        "n_requests": len(cache), "ctc_files": CTC_FILES, "system_prompt": system(a.keep_words),
        "shots_source": f"train only, round-robin over grammar_group_id, thai_refs[{a.ref_index}]",
        "shots": shots}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", meta.relative_to(BASE.parent))


if __name__ == "__main__":
    main()
