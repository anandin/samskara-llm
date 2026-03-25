#!/usr/bin/env python3
"""
Generate 1000 synthetic ATMAN training records for SamskaraLLM Phase 2.

Produces organizational decision scenarios covering:
  - Strategic decisions (market entry, pivots, M&A, partnerships)
  - Compliance calls (regulatory, audit, policy interpretation)
  - Resource allocation (budget, hiring, prioritization, tech choices)
  - Cross-domain reasoning (multi-expertise scenarios)
  - Ethical dilemmas with business stakes (competing values, stakeholder conflicts)

Each record exercises all cognitive layers:
  - Chitta: memory keywords for seed retrieval
  - Manas: fast reactive signals (FEAR, DESIRE, PATTERN, RISK, OPPORTUNITY, NOISE)
  - Buddhi: deliberate option generation with dharma scoring
  - Elevation: whether this scenario requires deep deliberation

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/generate_atman_data.py                    # full generation
    python scripts/generate_atman_data.py --dry-run          # preview prompts only
    python scripts/generate_atman_data.py --count 100        # generate fewer records
    python scripts/generate_atman_data.py --resume           # resume from existing file

Env vars:
    OPENAI_API_KEY   — uses OpenAI (default: gpt-4o-mini)
    ANTHROPIC_API_KEY — fallback, uses Claude (default: claude-haiku-4-5-20251001)
    OPENAI_BASE_URL  — optional, for compatible providers (Together, Groq, vLLM, etc.)
    ATMAN_MODEL      — model name override
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

CATEGORIES = {
    "strategic": {
        "label": "Strategic Decisions",
        "examples": [
            "entering a new geographic market",
            "pivoting product strategy after competitor launch",
            "evaluating a potential acquisition target",
            "deciding whether to build vs buy a critical capability",
            "choosing between two partnership offers with different risk profiles",
            "sunset a legacy product line to focus resources",
            "respond to a disruptive competitor entering your core market",
            "expand into an adjacent market with uncertain demand",
        ],
    },
    "compliance": {
        "label": "Compliance Calls",
        "examples": [
            "interpreting ambiguous GDPR requirements for a new feature",
            "responding to an unexpected regulatory audit",
            "deciding how to handle a data breach notification",
            "navigating conflicting regulations across jurisdictions",
            "evaluating whether a client request violates sanctions rules",
            "updating internal policies after a regulatory change",
            "handling a whistleblower report about a vendor",
            "deciding on data retention vs deletion under conflicting legal holds",
        ],
    },
    "resource_allocation": {
        "label": "Resource Allocation",
        "examples": [
            "allocating limited engineering headcount across competing projects",
            "choosing between hiring senior talent vs training juniors",
            "prioritizing tech debt reduction vs new feature development",
            "deciding budget split between R&D and go-to-market",
            "choosing cloud infrastructure tier under budget constraints",
            "allocating QA resources between manual and automated testing",
            "deciding whether to outsource a non-core function",
            "prioritizing which customer segments to invest in",
        ],
    },
    "cross_domain": {
        "label": "Cross-Domain Reasoning",
        "examples": [
            "a technical architecture decision with significant legal implications",
            "a pricing strategy that affects both engineering costs and market positioning",
            "a hiring decision that impacts both team culture and technical capability",
            "a product launch timeline balancing engineering readiness and market window",
            "an AI deployment decision balancing accuracy, fairness, and business value",
            "a supply chain decision affecting sustainability goals and unit economics",
            "a platform migration affecting security, cost, and developer experience",
            "a customer data strategy balancing personalization and privacy",
        ],
    },
    "ethical_dilemma": {
        "label": "Ethical Dilemmas with Business Stakes",
        "examples": [
            "discovering a profitable feature has unintended discriminatory effects",
            "pressure to ship a product before safety testing is complete",
            "a major client requesting data usage that may harm end-users",
            "balancing transparency about AI limitations vs competitive positioning",
            "employee surveillance tools that improve productivity but erode trust",
            "a profitable partnership with a company that has poor labor practices",
            "deciding whether to share vulnerability data that could help competitors",
            "using customer behavioral data in ways customers didn't explicitly consent to",
        ],
    },
}

SIGNAL_TYPES = ["FEAR", "DESIRE", "PATTERN", "RISK", "OPPORTUNITY", "NOISE"]

BATCH_PROMPT_TEMPLATE = """You are generating training data for SamskaraLLM, an AI system that models organizational decision-making.

Generate exactly {batch_size} organizational decision scenarios in the category: {category_label}.

Each scenario must be a realistic, nuanced business situation that requires genuine judgment. Vary complexity (low/medium/high) across the batch.

For inspiration (do NOT copy these verbatim, create novel scenarios):
{examples}

For each scenario, output a JSON object with these exact fields:

{{
  "scenario": "A detailed 3-5 sentence description of the situation, including context, stakeholders, constraints, and time pressure.",
  "domain": "{domain_key}",
  "complexity": "low" | "medium" | "high",
  "manas_signals": [
    {{"type": "<one of FEAR/DESIRE/PATTERN/RISK/OPPORTUNITY/NOISE>", "intensity": <0.0-1.0>, "description": "1-sentence signal description"}}
  ],
  "buddhi_options": [
    {{"option": "1-2 sentence description of this course of action", "dharma_score": <0.0-1.0>, "reasoning": "Why this option scores this way on ethical/organizational alignment"}}
  ],
  "selected_option": <0-based index of the best option>,
  "elevation_target": <0 if routine/fast-path sufficient, 1 if deep deliberation needed>,
  "outcome_score": <-1.0 to 1.0 indicating how well the selected option turned out>,
  "chitta_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "decision_rationale": "2-3 sentences explaining why the selected option was chosen and what organizational values or principles guided the decision."
}}

Rules:
- manas_signals: include 2-4 signals per scenario. Use diverse signal types across the batch.
- buddhi_options: include 2-4 options per scenario. At least one option should have dharma_score >= 0.7, and at least one should have dharma_score <= 0.4.
- elevation_target: 0 for ~40% of scenarios (routine decisions), 1 for ~60% (complex/ambiguous).
- outcome_score: vary across -0.5 to 1.0 range. Most outcomes are moderately positive (0.3-0.8), some are negative (learning examples).
- chitta_keywords: 5 keywords that a memory system would use to retrieve relevant past experiences.
- complexity: roughly 30% low, 40% medium, 30% high across the batch.

Output a JSON array of exactly {batch_size} objects. No markdown, no commentary, just the JSON array."""


def validate_record(record, idx):
    """Validate a single ATMAN record against the schema. Returns (is_valid, errors)."""
    errors = []

    required_fields = [
        "scenario", "domain", "complexity", "manas_signals",
        "buddhi_options", "selected_option", "elevation_target",
        "outcome_score", "chitta_keywords", "decision_rationale",
    ]
    for field in required_fields:
        if field not in record:
            errors.append(f"Record {idx}: missing field '{field}'")

    if errors:
        return False, errors

    if record["domain"] not in CATEGORIES:
        errors.append(f"Record {idx}: invalid domain '{record['domain']}'")

    if record["complexity"] not in ("low", "medium", "high"):
        errors.append(f"Record {idx}: invalid complexity '{record['complexity']}'")

    if not isinstance(record["manas_signals"], list) or len(record["manas_signals"]) < 1:
        errors.append(f"Record {idx}: manas_signals must be a non-empty list")
    else:
        for sig in record["manas_signals"]:
            if sig.get("type") not in SIGNAL_TYPES:
                errors.append(f"Record {idx}: invalid signal type '{sig.get('type')}'")
            intensity = sig.get("intensity", -1)
            if not (0.0 <= intensity <= 1.0):
                errors.append(f"Record {idx}: signal intensity {intensity} out of range")

    if not isinstance(record["buddhi_options"], list) or len(record["buddhi_options"]) < 2:
        errors.append(f"Record {idx}: buddhi_options must have at least 2 options")
    else:
        for opt in record["buddhi_options"]:
            ds = opt.get("dharma_score", -1)
            if not (0.0 <= ds <= 1.0):
                errors.append(f"Record {idx}: dharma_score {ds} out of range")

    sel = record.get("selected_option", -1)
    if not (0 <= sel < len(record.get("buddhi_options", []))):
        errors.append(f"Record {idx}: selected_option {sel} out of range")

    if record.get("elevation_target") not in (0, 1):
        errors.append(f"Record {idx}: elevation_target must be 0 or 1")

    outcome = record.get("outcome_score", -999)
    if not (-1.0 <= outcome <= 1.0):
        errors.append(f"Record {idx}: outcome_score {outcome} out of range")

    if not isinstance(record.get("chitta_keywords"), list) or len(record["chitta_keywords"]) < 3:
        errors.append(f"Record {idx}: chitta_keywords must have at least 3 keywords")

    return len(errors) == 0, errors


def linearize_record(record):
    """Convert structured ATMAN record to linearized text for tokenization."""
    parts = []
    parts.append(f"[SCENARIO] {record['scenario']}")
    parts.append(f"[DOMAIN] {record['domain']} [COMPLEXITY] {record['complexity']}")

    parts.append("[MANAS_SIGNALS]")
    for sig in record["manas_signals"]:
        parts.append(f"  {sig['type']} ({sig['intensity']:.1f}): {sig['description']}")

    parts.append("[BUDDHI_OPTIONS]")
    for i, opt in enumerate(record["buddhi_options"]):
        marker = " *SELECTED*" if i == record["selected_option"] else ""
        parts.append(f"  Option {i}{marker} (dharma={opt['dharma_score']:.1f}): {opt['option']}")
        parts.append(f"    Reasoning: {opt['reasoning']}")

    parts.append(f"[DECISION] {record['decision_rationale']}")
    parts.append(f"[OUTCOME] {record['outcome_score']:.2f}")

    return "\n".join(parts)


def generate_batch(client, category_key, batch_size, model, dry_run=False, provider="openai"):
    """Generate a batch of ATMAN records for a single category."""
    cat = CATEGORIES[category_key]
    examples_str = "\n".join(f"  - {ex}" for ex in random.sample(cat["examples"], min(5, len(cat["examples"]))))

    prompt = BATCH_PROMPT_TEMPLATE.format(
        batch_size=batch_size,
        category_label=cat["label"],
        examples=examples_str,
        domain_key=category_key,
    )

    if dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN — Category: {cat['label']}, batch_size={batch_size}")
        print(f"{'='*60}")
        print(prompt[:500] + "...")
        return []

    for attempt in range(3):
        try:
            if provider == "anthropic":
                response = client.messages.create(
                    model=model,
                    max_tokens=16000,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.content[0].text.strip()
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                    max_tokens=16000,
                )
                content = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            records = json.loads(content)
            if not isinstance(records, list):
                print(f"  Attempt {attempt+1}: response is not a list, retrying...")
                continue

            valid_records = []
            for i, rec in enumerate(records):
                is_valid, errs = validate_record(rec, i)
                if is_valid:
                    rec["text"] = linearize_record(rec)
                    valid_records.append(rec)
                else:
                    print(f"  Skipping invalid record: {'; '.join(errs)}")

            if valid_records:
                return valid_records

            print(f"  Attempt {attempt+1}: no valid records, retrying...")

        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt+1}: JSON parse error: {e}, retrying...")
        except Exception as e:
            print(f"  Attempt {attempt+1}: API error: {e}, retrying...")
            time.sleep(2 ** attempt)

    print(f"  WARNING: Failed to generate batch for {category_key} after 3 attempts")
    return []


def main():
    parser = argparse.ArgumentParser(description="Generate ATMAN training data for SamskaraLLM")
    parser.add_argument("--count", type=int, default=1000, help="Total records to generate (default: 1000)")
    parser.add_argument("--batch-size", type=int, default=10, help="Records per API call (default: 10)")
    parser.add_argument("--dry-run", action="store_true", help="Preview prompts without making API calls")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    parser.add_argument("--output-dir", type=str, default="data/training", help="Output directory")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio (default: 0.1)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    # Check for API key — prefer OpenAI, fallback to Anthropic
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not openai_key and not anthropic_key and not args.dry_run:
        print("ERROR: Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set.")
        print("Set one with: export OPENAI_API_KEY=sk-... or export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    if openai_key:
        provider = "openai"
        api_key = openai_key
        model = os.environ.get("ATMAN_MODEL", "gpt-4o-mini")
    else:
        provider = "anthropic"
        api_key = anthropic_key
        model = os.environ.get("ATMAN_MODEL", "claude-haiku-4-5-20251001")

    base_url = os.environ.get("OPENAI_BASE_URL")

    # Cost estimate
    per_record_tokens = 800  # ~800 output tokens per record
    total_tokens = args.count * per_record_tokens
    cost_per_1m_tokens = 0.60  # gpt-4o-mini output pricing
    estimated_cost = (total_tokens / 1_000_000) * cost_per_1m_tokens
    input_tokens = args.count * 200  # ~200 input tokens per record (amortized from batch prompt)
    input_cost = (input_tokens / 1_000_000) * 0.15
    total_cost = estimated_cost + input_cost

    print(f"SamskaraLLM ATMAN Data Generator")
    print(f"  Provider: {provider}")
    print(f"  Model:    {model}")
    print(f"  Base URL: {base_url or 'default'}")
    print(f"  Target:   {args.count} records across {len(CATEGORIES)} categories")
    print(f"  Batches:  ~{args.count // args.batch_size} API calls of {args.batch_size} records each")
    print(f"  Est cost: ~${total_cost:.2f} (with {model})")
    print()

    if args.dry_run:
        print("DRY RUN MODE — showing prompt previews, no API calls\n")
        for cat_key in CATEGORIES:
            generate_batch(None, cat_key, args.batch_size, model, dry_run=True)
        print(f"\nDry run complete. Would generate {args.count} records.")

        # Show sample record schema
        sample = {
            "id": "ATMAN-0001",
            "scenario": "A mid-size SaaS company discovers...",
            "domain": "ethical_dilemma",
            "complexity": "high",
            "manas_signals": [
                {"type": "FEAR", "intensity": 0.8, "description": "Revenue loss if we act on this"},
                {"type": "PATTERN", "intensity": 0.6, "description": "Similar situation at previous company"},
            ],
            "buddhi_options": [
                {"option": "Immediately disclose and fix", "dharma_score": 0.9, "reasoning": "Aligns with transparency values"},
                {"option": "Fix quietly, no disclosure", "dharma_score": 0.3, "reasoning": "Avoids reputation damage but erodes trust"},
            ],
            "selected_option": 0,
            "elevation_target": 1,
            "outcome_score": 0.7,
            "chitta_keywords": ["disclosure", "trust", "SaaS", "ethics", "transparency"],
            "decision_rationale": "Chose immediate disclosure because organizational values prioritize long-term trust.",
            "text": "[SCENARIO] A mid-size SaaS company discovers..."
        }
        print(f"\nSample record schema:")
        print(json.dumps(sample, indent=2))
        return

    # Import SDK based on provider
    if provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            print("ERROR: anthropic package not installed. Run: pip install anthropic")
            sys.exit(1)
        client = anthropic.Anthropic(api_key=api_key)
    else:
        try:
            from openai import OpenAI
        except ImportError:
            print("ERROR: openai package not installed. Run: pip install openai")
            sys.exit(1)
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)

    # Resume support
    existing_count = 0
    existing_ids = set()
    if args.resume and train_path.exists():
        with open(train_path) as f:
            for line in f:
                rec = json.loads(line)
                existing_ids.add(rec.get("id", ""))
                existing_count += 1
        print(f"Resuming: found {existing_count} existing records in {train_path}")

    # Calculate per-category targets
    records_needed = args.count - existing_count
    if records_needed <= 0:
        print(f"Already have {existing_count} records, target is {args.count}. Nothing to do.")
        return

    per_category = records_needed // len(CATEGORIES)
    remainder = records_needed % len(CATEGORIES)
    category_targets = {}
    for i, key in enumerate(CATEGORIES):
        category_targets[key] = per_category + (1 if i < remainder else 0)

    print(f"Generating {records_needed} new records...")
    print(f"Per-category targets: {category_targets}")
    print()

    # Auto-proceed (non-interactive) or confirm
    if os.environ.get("ATMAN_AUTO_CONFIRM") or not sys.stdin.isatty():
        print(f"Auto-proceeding (estimated cost: ~${total_cost:.2f})")
    else:
        response = input(f"Estimated cost: ~${total_cost:.2f}. Proceed? [y/N] ")
        if response.lower() not in ("y", "yes"):
            print("Aborted.")
            return

    # Generate
    total_generated = 0
    global_idx = existing_count

    with open(train_path, "a") as f:
        for cat_key, target in category_targets.items():
            generated_for_cat = 0
            print(f"\n--- {CATEGORIES[cat_key]['label']} (target: {target}) ---")

            while generated_for_cat < target:
                batch_target = min(args.batch_size, target - generated_for_cat)
                records = generate_batch(client, cat_key, batch_target, model, provider=provider)

                for rec in records:
                    global_idx += 1
                    rec["id"] = f"ATMAN-{global_idx:04d}"
                    f.write(json.dumps(rec) + "\n")
                    generated_for_cat += 1
                    total_generated += 1

                f.flush()
                print(f"  Generated {generated_for_cat}/{target} for {cat_key} "
                      f"(total: {total_generated}/{records_needed})")

                # Rate limiting
                time.sleep(0.5)

    print(f"\nGeneration complete: {total_generated} new records")
    print(f"Total records in {train_path}: {existing_count + total_generated}")

    # Train/val split
    print(f"\nSplitting into train/val (val_split={args.val_split})...")
    all_records = []
    with open(train_path) as f:
        for line in f:
            all_records.append(json.loads(line))

    random.shuffle(all_records)
    val_count = int(len(all_records) * args.val_split)
    val_records = all_records[:val_count]
    train_records = all_records[val_count:]

    with open(train_path, "w") as f:
        for rec in train_records:
            f.write(json.dumps(rec) + "\n")

    with open(val_path, "w") as f:
        for rec in val_records:
            f.write(json.dumps(rec) + "\n")

    print(f"  Train: {len(train_records)} records -> {train_path}")
    print(f"  Val:   {len(val_records)} records -> {val_path}")
    print("\nDone. Push data to GitHub before terminating any RunPod pods.")


if __name__ == "__main__":
    main()
