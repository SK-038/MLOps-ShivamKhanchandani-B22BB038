import sys
from pathlib import Path
from sacrebleu.metrics import BLEU, CHRF


def load_lines(filepath: str) -> list:
    return [
        line.strip()
        for line in Path(filepath).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate(hypothesis_path: str, reference_path: str) -> float:
    hypotheses = load_lines(hypothesis_path)
    references  = load_lines(reference_path)

    if len(hypotheses) != len(references):
        print(
            f"Warning: {hypothesis_path} has {len(hypotheses)} lines, "
            f"{reference_path} has {len(references)} lines. Truncating to shorter.",
            file=sys.stderr,
        )
        min_len     = min(len(hypotheses), len(references))
        hypotheses  = hypotheses[:min_len]
        references  = references[:min_len]

    bleu = BLEU(effective_order=True)
    chrf = CHRF()

    bleu_score = bleu.corpus_score(hypotheses, [references])
    chrf_score = chrf.corpus_score(hypotheses, [references])

    print(f"Hypothesis : {hypothesis_path}")
    print(f"Reference  : {reference_path}")
    print(f"Sentences  : {len(hypotheses)}")
    print(f"BLEU Score : {bleu_score}")
    print(f"chrF Score : {chrf_score}")

    return bleu_score.score


if __name__ == "__main__":
    evaluate("output.txt", "reference.txt")