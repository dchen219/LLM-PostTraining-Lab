"""Stage 0: build verified data, splits, matched subsets, hashes and contamination report (CPU only).

Originals are read only. Refuses to overwrite: --out must not already contain a manifest.
Usage: python scripts/stage0_build_data.py --src <sft-reason dir> --gsm8k-train <parquet> --gsm8k-test <jsonl>
           --audit <recovery_audit.jsonl> --out data/stage0
"""
import argparse, collections, hashlib, json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from posttrain.verifiers.drgrpo_grader import r1_zero_reward_fn  # noqa: E402

SEED = 2026
SIZES = [128, 256, 512, 1024, 2048]
DATA_SEEDS = [0, 1, 2]


def load(p):
    t = open(p, encoding="utf-8").read()
    return json.loads(t) if t.lstrip()[0] == "[" else [json.loads(x) for x in t.splitlines() if x.strip()]


def sid(r):
    return hashlib.sha256((r["problem"] + "\0" + r["reasoning_trace"]).encode()).hexdigest()[:16]


def pid(q):
    return hashlib.sha256(re.sub(r"\s+", " ", q).strip().encode()).hexdigest()[:16]


norm = lambda s: re.sub(r"\W+", "", s.lower())
words = lambda s: re.findall(r"\w+", s.lower())
grams = lambda s, n: {tuple(w[i:i + n]) for w in [words(s)] for i in range(len(w) - n + 1)}


def dump(path, rows):
    assert not os.path.exists(path), f"refusing to overwrite {path}"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def overlap(a, b, n=13, j=0.6):
    """Pairs between question lists a and b: exact-normalised, shared n-gram, fuzzy 5-gram Jaccard >= j."""
    nb = {}
    for i, q in enumerate(b):
        nb.setdefault(norm(q), i)
    exact = {i for i, q in enumerate(a) if norm(q) in nb}
    idx = collections.defaultdict(set)
    for i, q in enumerate(b):
        for g in grams(q, n):
            idx[g].add(i)
    ng = {i for i, q in enumerate(a) if any(g in idx for g in grams(q, n))}
    idx5 = collections.defaultdict(set); gb = [grams(q, 5) for q in b]
    for i, gs in enumerate(gb):
        for g in gs:
            idx5[g].add(i)
    fuzzy = set(); pairs = []
    for i, q in enumerate(a):
        ga = grams(q, 5); c = collections.Counter()
        for g in ga:
            for k in idx5.get(g, ()):
                c[k] += 1
        for k, v in c.most_common(3):
            jac = v / max(1, len(ga | gb[k]))
            if jac >= j:
                fuzzy.add(i); pairs.append((round(jac, 3), q[:110], b[k][:110])); break
    return {"exact": exact, "ngram": ng, "fuzzy": fuzzy, "pairs": sorted(pairs, reverse=True)}


def main():
    ap = argparse.ArgumentParser()
    for x in ("src", "gsm8k-train", "gsm8k-test", "audit", "out"):
        ap.add_argument("--" + x, required=True)
    a = ap.parse_args(); out = a.out
    assert not os.path.exists(f"{out}/MANIFEST.json"), "output already built; use a new directory"
    os.makedirs(f"{out}/subsets", exist_ok=True)

    raw = load(f"{a.src}/sft_gpt-oss-120b.jsonl"); fil = load(f"{a.src}/sft_gpt-oss-120b_filtered.jsonl")
    val = load(f"{a.src}/val.jsonl"); pool = load(f"{a.src}/train.jsonl")
    recovered = {x["id"]: x["tier"] for x in load(a.audit) if x["recovered"]}
    exact_ids = {sid(r) for r in fil}
    rows = []
    for r in raw:
        i = sid(r); fm = r1_zero_reward_fn(r["reasoning_trace"], r["expected_answer"])
        rows.append(dict(id=i, problem_id=pid(r["problem"]), problem=r["problem"], reasoning_trace=r["reasoning_trace"],
                         extracted_answer=r["extracted_answer"], expected_answer=r["expected_answer"],
                         exact_match=i in exact_ids, reverified=i in exact_ids or i in recovered,
                         recovery_tier=recovered.get(i), strict_reward=fm["reward"], strict_format=fm["format_reward"]))
    rev = [r for r in rows if r["reverified"]]
    assert len(raw) == 4836 and len(fil) == 3496 and len(rev) == 4545, (len(raw), len(fil), len(rev))
    assert len({r["id"] for r in rows}) == len(rows)
    first, dup_ids, dup_pairs = {}, set(), []
    for r in rows:  # whitespace/punctuation/case-insensitive duplicate problems
        k = norm(r["problem"])
        if k in first:
            dup_ids.add(r["id"]); dup_pairs.append(dict(dup=r["id"], first=first[k]["id"], same_gold=r["expected_answer"] == first[k]["expected_answer"],
                             first_problem=first[k]["problem"][:120], dup_problem=r["problem"][:120]))
        else:
            first[k] = r
    dump(f"{out}/sft_all_labeled_4836.jsonl", rows)
    dump(f"{out}/sft_reverified_4545.jsonl", rev)
    dump(f"{out}/sft_exactmatch_3496.jsonl", [r for r in rows if r["exact_match"]])

    # ---- evaluation splits ----
    import pyarrow.parquet as pq
    gtr = pq.read_table(a.gsm8k_train).to_pylist(); gte = load(a.gsm8k_test)
    ans = lambda t: re.search(r"####\s*(.+)\s*$", t.strip()).group(1).strip()
    g = lambda split, r: dict(id=f"gsm8k-{split}-" + pid(r["question"]), question=r["question"], answer=ans(r["answer"]))
    gtr = [g("train", r) for r in gtr]; gte = [g("test", r) for r in gte]
    rng = random.Random(SEED); order = list(range(len(gtr))); rng.shuffle(order)
    gdev = [gtr[i] for i in sorted(order[:1000])]
    mk = lambda split, r: dict(id=f"math-{split}-" + pid(r["problem"]), question=r["problem"], answer=str(r["expected_answer"]))
    mfinal = [mk("val", r) for r in val if str(r["expected_answer"]) not in ("None", "")]
    traced = {r["problem_id"] for r in rows}
    untraced = [r for r in pool if pid(r["problem"]) not in traced and str(r["expected_answer"]) not in ("None", "")]

    # ---- contamination ----
    Q = {"sft_4836": [r["problem"] for r in rows], "rl_pool": [r["problem"] for r in pool],
         "gsm8k_dev": [r["question"] for r in gdev], "gsm8k_test": [r["question"] for r in gte],
         "math_final": [r["question"] for r in mfinal]}
    cont = {}; keep = {}
    for x in ("sft_4836", "rl_pool"):
        for y in ("gsm8k_dev", "gsm8k_test", "math_final"):
            o = overlap(Q[x], Q[y]); keep[(x, y)] = o
            cont[f"{x} vs {y}"] = {k: len(o[k]) for k in ("exact", "ngram", "fuzzy")} | {"top_pairs": o["pairs"][:5]}
    o = overlap(Q["gsm8k_dev"], Q["gsm8k_test"]); cont["gsm8k_dev vs gsm8k_test"] = {k: len(o[k]) for k in ("exact", "ngram", "fuzzy")} | {"top_pairs": o["pairs"][:5]}
    for x in Q:
        c = collections.Counter(norm(q) for q in Q[x]); cont[f"{x} internal exact duplicates"] = sum(v - 1 for v in c.values() if v > 1)
    json.dump(cont, open(f"{out}/contamination.json", "w"), indent=1, ensure_ascii=False)

    # flag (not delete) training problems that are exact or fuzzy near-duplicates of any evaluation set
    flag_sft, flag_pool = set(), set()
    for y in ("gsm8k_dev", "gsm8k_test", "math_final"):
        flag_sft |= keep[("sft_4836", y)]["exact"] | keep[("sft_4836", y)]["fuzzy"]
        flag_pool |= keep[("rl_pool", y)]["exact"] | keep[("rl_pool", y)]["fuzzy"]
    clean_sft = {rows[i]["id"] for i in range(len(rows)) if i not in flag_sft} - dup_ids
    flagged_p = {pid(pool[i]["problem"]) for i in flag_pool}
    rng = random.Random(SEED + 1); u = [r for r in untraced if pid(r["problem"]) not in flagged_p]; rng.shuffle(u)
    mdev = [mk("dev", r) for r in u[:500]]
    devp = {pid(r["question"]) for r in mdev}
    seen, rl = set(), []
    for r in pool:
        p = pid(r["problem"])
        if p in devp or p in flagged_p or p in seen or str(r["expected_answer"]) in ("None", ""):
            continue
        seen.add(p); rl.append(dict(id="rl-" + p, question=r["problem"], answer=str(r["expected_answer"])))
    for n, x in (("eval_gsm8k_dev_1000", gdev), ("eval_gsm8k_test_final", gte), ("eval_math_dev_500", mdev),
                 ("eval_math_val_final", mfinal), ("rl_train_pool", rl)):
        dump(f"{out}/{n}.jsonl", x)

    # ---- matched, nested subsets ----
    arms = {"raw": lambda r: True, "exact": lambda r: r["exact_match"], "reverified": lambda r: r["reverified"]}
    sub = {}
    for s in DATA_SEEDS:
        key = lambda r: hashlib.sha256(f"{SEED}:{s}:{r['id']}".encode()).hexdigest()
        ordered = sorted((r for r in rows if r["id"] in clean_sft), key=key)
        for arm, f in arms.items():
            elig = [r["id"] for r in ordered if f(r)]
            sizes = SIZES if arm == "reverified" else SIZES if arm == "exact" else [512]
            for n in sizes:
                sub[f"{arm}_n{n}_s{s}"] = elig[:n]
            if s == 0:
                sub[f"{arm}_full"] = elig
        # size control: same N as the exact-match full set, drawn from the re-verified pool
        sub[f"reverified_sizecontrol_s{s}"] = [r["id"] for r in ordered if r["reverified"]][:len([r for r in ordered if r["exact_match"]])]
    for n, ids in sub.items():
        json.dump({"name": n, "n": len(ids), "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(), "ids": ids},
                  open(f"{out}/subsets/{n}.json", "w"))
    ov = {f"s{s}": {f"{x}&{y}": len(set(sub[f"{x}_n512_s{s}"]) & set(sub[f"{y}_n512_s{s}"]))
                    for x, y in (("raw", "exact"), ("raw", "reverified"), ("exact", "reverified"))} for s in DATA_SEEDS}
    verified_in_raw512 = {f"s{s}": sum(1 for i in sub[f"raw_n512_s{s}"] if i in {r["id"] for r in rev}) for s in DATA_SEEDS}

    man = {"seed": SEED, "counts": {"raw": len(rows), "exact": len(fil), "reverified": len(rev),
           "reverified_and_strict_format": sum(r["strict_reward"] == 1 for r in rev), "gsm8k_dev": len(gdev), "gsm8k_test": len(gte),
           "math_dev": len(mdev), "math_final": len(mfinal), "rl_pool": len(rl), "rl_pool_source": len(pool), "untraced_pool": len(untraced)},
           "duplicate_problems_in_sft": {"n": len(dup_ids), "pairs": dup_pairs[:12]}, "clean_sft_unique": len(clean_sft),
           "contamination_flagged": {"sft": len(flag_sft), "rl_pool": len(flag_pool)},
           "subset_overlap_512": ov, "reverified_inside_raw512": verified_in_raw512,
           "source_sha256": {os.path.basename(p): sha(p) for p in (f"{a.src}/sft_gpt-oss-120b.jsonl", f"{a.src}/sft_gpt-oss-120b_filtered.jsonl",
                             f"{a.src}/val.jsonl", f"{a.src}/train.jsonl", a.gsm8k_train, a.gsm8k_test)},
           "output_sha256": {f: sha(f"{out}/{f}") for f in sorted(os.listdir(out)) if f.endswith(".jsonl")},
           "subsets": {n: {"n": len(i)} for n, i in sub.items()}}
    json.dump(man, open(f"{out}/MANIFEST.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in man.items() if k != "subsets"}, indent=1)); print(json.dumps(cont, indent=1, ensure_ascii=False)[:7000])


if __name__ == "__main__":
    main()
