import argparse
import sys
import torch
from pathlib import Path
from transformers import MarianMTModel, MarianTokenizer
from sacrebleu.metrics import BLEU, CHRF

MODEL_NAME = "Helsinki-NLP/opus-mt-bn-en"


def load_model(model_name: str = MODEL_NAME):
    print(f"Loading model: {model_name}")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def translate_sentences(sentences: list, tokenizer, model, batch_size: int = 8) -> list:
    translations = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            outputs = model.generate(**inputs, num_beams=4, max_length=512)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translations.extend(decoded)
            print(f"  Translated {min(i + batch_size, len(sentences))}/{len(sentences)} sentences", end="\r")
    print()
    return translations


def read_lines(path: str) -> list:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def evaluate(hypotheses: list, references: list):
    bleu = BLEU(effective_order=True)
    chrf = CHRF()
    bleu_score = bleu.corpus_score(hypotheses, [references])
    chrf_score = chrf.corpus_score(hypotheses, [references])
    return bleu_score, chrf_score


def main():
    parser = argparse.ArgumentParser(description="Translate Bengali to English and evaluate with BLEU/chrF")
    parser.add_argument("--input",      default="input.txt",     help="Input file (Bengali)")
    parser.add_argument("--reference",  default="reference.txt", help="Reference translations (English)")
    parser.add_argument("--output",     default="output.txt",    help="Where to save translated output")
    parser.add_argument("--model",      default=MODEL_NAME,      help="HuggingFace model name/path")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    source_sentences = read_lines(args.input)
    print(f"Loaded {len(source_sentences)} source sentences from '{args.input}'")

    tokenizer, model = load_model(args.model)

    print("Translating...")
    translations = translate_sentences(source_sentences, tokenizer, model, batch_size=args.batch_size)

    Path(args.output).write_text("\n".join(translations) + "\n", encoding="utf-8")
    print(f"Saved {len(translations)} translations to '{args.output}'")

    print("\n--- Translation Preview ---")
    for i, (src, hyp) in enumerate(zip(source_sentences, translations), 1):
        print(f"[{i:02d}] BN : {src}")
        print(f"     EN : {hyp}")
        print()

    if Path(args.reference).exists():
        references = read_lines(args.reference)
        if len(references) != len(translations):
            print(
                f"Warning: reference has {len(references)} lines but translations have "
                f"{len(translations)} lines — skipping evaluation.",
                file=sys.stderr,
            )
        else:
            bleu_score, chrf_score = evaluate(translations, references)
            print("--- Evaluation Metrics ---")
            print(f"BLEU : {bleu_score}")
            print(f"chrF : {chrf_score}")

            metrics_path = Path(args.output).with_suffix(".metrics.txt")
            metrics_path.write_text(
                f"BLEU : {bleu_score}\nchrF : {chrf_score}\n", encoding="utf-8"
            )
            print(f"Metrics saved to '{metrics_path}'")
    else:
        print(f"Reference file '{args.reference}' not found — skipping evaluation.")


if __name__ == "__main__":
    main()