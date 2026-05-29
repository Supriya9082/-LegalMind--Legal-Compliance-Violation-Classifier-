"""
Script: Generate synthetic labeled training data using Claude API.

This is the practical solution to the labeled-data problem.
Uses the Anthropic API to generate realistic SEBI/GDPR violation
and compliance examples based on templates from real enforcement orders.

Usage:
  pip install anthropic
  export ANTHROPIC_API_KEY=your_key_here
  python scripts/generate_synthetic_data.py --n 500 --output data/finetune/labeled.json
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import random
import argparse
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

VIOLATION_PROMPTS = [
    "Generate a realistic 2-3 sentence legal text snippet describing a SEBI regulation violation involving insider trading.",
    "Generate a 2-3 sentence text about a company failing to disclose material information to SEBI.",
    "Generate a 2-3 sentence text about a broker violating GDPR by sharing client data without consent.",
    "Generate a 2-3 sentence SEBI enforcement scenario where a fund manager manipulated NAV.",
    "Generate a 2-3 sentence GDPR violation where a company processed personal data without legal basis.",
    "Generate a 2-3 sentence scenario of front-running by a securities broker.",
    "Generate a 2-3 sentence SEBI violation involving circular trading to inflate volumes.",
    "Generate a 2-3 sentence GDPR breach involving inadequate data security leading to personal data exposure.",
]

COMPLIANT_PROMPTS = [
    "Generate a 2-3 sentence legal text describing a company properly disclosing quarterly results to SEBI.",
    "Generate a 2-3 sentence text about a broker maintaining compliant KYC records per SEBI norms.",
    "Generate a 2-3 sentence description of a company obtaining valid GDPR consent from users.",
    "Generate a 2-3 sentence text about a fund correctly following SEBI portfolio disclosure requirements.",
    "Generate a 2-3 sentence description of GDPR-compliant data retention and deletion practices.",
    "Generate a 2-3 sentence text about a listed company following SEBI insider trading prevention protocols.",
    "Generate a 2-3 sentence description of a broker properly segregating client funds per SEBI rules.",
    "Generate a 2-3 sentence GDPR-compliant privacy notice with clear purpose and data minimisation.",
]


def generate_with_anthropic(prompt: str, api_key: str) -> str:
    """Call Anthropic API to generate a single sample."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model      = "claude-3-haiku-20240307",
            max_tokens = 200,
            messages   = [{
                "role":    "user",
                "content": prompt + " Output only the text snippet, no labels or explanations.",
            }],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        logger.warning(f"API error: {e}")
        return ""


def generate_rule_based(label: int) -> str:
    """
    Fallback: template-based generation (no API key needed).
    Less diverse but functional for smoke-testing.
    """
    violations = [
        "The accused entity executed a series of synchronized buy and sell orders in scrip XYZ "
        "to create a false impression of trading volume, thereby violating SEBI (PFUTP) Regulations.",
        "The respondent, being in possession of unpublished price-sensitive information, traded in "
        "the shares of the company through a connected person, in contravention of PIT Regulations.",
        "The data controller processed special category personal data of EU residents without obtaining "
        "explicit consent, in violation of GDPR Article 9.",
        "Personal data of approximately 15,000 data subjects was retained beyond the stated retention "
        "period without a lawful basis, breaching GDPR Article 5(1)(e).",
        "The broker commingled client funds with proprietary funds, failing to maintain the required "
        "segregation as mandated by SEBI Broker Regulations.",
    ]
    compliant = [
        "The listed entity disclosed all material price-sensitive information on the stock exchange "
        "platform within 24 hours of the board meeting, in compliance with SEBI LODR Regulations.",
        "The mutual fund scheme's portfolio was disclosed on the AMFI website within the prescribed "
        "timelines, fulfilling all SEBI disclosure requirements.",
        "The data controller obtained freely given, specific, informed and unambiguous consent from "
        "all data subjects prior to processing their personal data, in line with GDPR Article 6.",
        "The company maintained a comprehensive Record of Processing Activities and appointed a Data "
        "Protection Officer as required under GDPR Article 37.",
        "KYC documentation for all clients was verified and updated within the period prescribed by "
        "SEBI, with no instances of non-compliant onboarding.",
    ]
    pool = violations if label == 1 else compliant
    return random.choice(pool)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int, default=500, help="Total samples to generate")
    parser.add_argument("--output",  default="data/finetune/labeled.json")
    parser.add_argument("--api-key", default=os.getenv("ANTHROPIC_API_KEY", ""),
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--rule-based", action="store_true",
                        help="Use rule-based generation (no API key needed)")
    args = parser.parse_args()

    use_api = bool(args.api_key) and not args.rule_based
    if use_api:
        logger.info(f"[DataGen] Using Anthropic API to generate {args.n} samples")
    else:
        logger.info(f"[DataGen] Using rule-based generation for {args.n} samples")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    samples  = []
    n_each   = args.n // 2  # 50/50 class balance

    # Generate violations (label=1)
    logger.info(f"[DataGen] Generating {n_each} violation samples...")
    for i in range(n_each):
        if use_api:
            prompt = random.choice(VIOLATION_PROMPTS)
            text   = generate_with_anthropic(prompt, args.api_key)
            time.sleep(0.5)   # rate limit
        else:
            text = generate_rule_based(1)
        if text:
            samples.append({"text": text, "label": 1})
        if (i+1) % 50 == 0:
            logger.info(f"  violations: {i+1}/{n_each}")

    # Generate compliant (label=0)
    logger.info(f"[DataGen] Generating {n_each} compliant samples...")
    for i in range(n_each):
        if use_api:
            prompt = random.choice(COMPLIANT_PROMPTS)
            text   = generate_with_anthropic(prompt, args.api_key)
            time.sleep(0.5)
        else:
            text = generate_rule_based(0)
        if text:
            samples.append({"text": text, "label": 0})
        if (i+1) % 50 == 0:
            logger.info(f"  compliant: {i+1}/{n_each}")

    random.shuffle(samples)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    logger.info(f"[DataGen] Saved {len(samples)} samples → {args.output}")
    logger.info(f"  violations : {sum(s['label']==1 for s in samples)}")
    logger.info(f"  compliant  : {sum(s['label']==0 for s in samples)}")


if __name__ == "__main__":
    main()
