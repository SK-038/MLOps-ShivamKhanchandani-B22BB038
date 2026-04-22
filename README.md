# MLOps Q1 - Bengali to English Translation Pipeline

This folder contains the MLOps Q1 assignment submission. The project translates Bengali text to English using a Hugging Face MarianMT model and evaluates the generated translations with BLEU and chrF metrics.

## Repository Details

- GitHub ID: `SK-038`
- Email: `b22bb038@iitj.ac.in`
- Repository: `MLOps-ShivamKhanchandani-B22BB038`
- Submission branch: `MLDLOps-Exam2026`
- Commit hash: `11fe5027c36d28df4248a5e09c98d0217f0503f8`
- Commit message: `Add mlops q1 submission`
- Author: `SK-038 <b22bb038@iitj.ac.in>`
- Committer: `SK-038 <b22bb038@iitj.ac.in>`

## Files Included

The following files are included in this submission:

```text
Dockerfile
evaluate.py
input.txt
output.metrics.txt
output.txt
reference.txt
requirements.txt
translate.py
README.md
```

## Project Description

The translation pipeline uses the pretrained Hugging Face model:

```text
Helsinki-NLP/opus-mt-bn-en
```

The workflow is:

1. Read Bengali input sentences from `input.txt`.
2. Translate the sentences into English using `translate.py`.
3. Save generated translations to `output.txt`.
4. Compare generated translations with references from `reference.txt`.
5. Compute BLEU and chrF scores using SacreBLEU.
6. Save metric results to `output.metrics.txt`.

## Requirements

Python dependencies are listed in `requirements.txt`:

```text
transformers==4.40.0
torch==2.2.2
sentencepiece==0.2.0
sacrebleu==2.4.0
numpy<2.0
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run translation and evaluation:

```bash
python translate.py
```

Run evaluation separately:

```bash
python evaluate.py
```

## Docker Usage

Build the Docker image:

```bash
docker build -t mlops-q1-translation .
```

Run the Docker container:

```bash
docker run --rm mlops-q1-translation
```

The container executes:

```bash
python translate.py && python evaluate.py
```

## File Descriptions

- `Dockerfile`: Defines the containerized environment for running the translation and evaluation pipeline.
- `requirements.txt`: Lists the Python packages required to run the project locally.
- `translate.py`: Loads the translation model, translates Bengali input sentences, writes output, and computes metrics when references are available.
- `evaluate.py`: Evaluates `output.txt` against `reference.txt` using BLEU and chrF.
- `input.txt`: Contains Bengali source sentences.
- `reference.txt`: Contains reference English translations.
- `output.txt`: Contains generated English translations.
- `output.metrics.txt`: Contains the BLEU and chrF metric results.

## Commit Information

Only the assignment files from the `mlops_q1` folder were added to the submission branch.

```text
Commit Hash : 11fe5027c36d28df4248a5e09c98d0217f0503f8
Branch      : MLDLOps-Exam2026
Author      : SK-038 <b22bb038@iitj.ac.in>
Committer   : SK-038 <b22bb038@iitj.ac.in>
Message     : Add mlops q1 submission
```

## Branch Submission

The assignment was committed on a branch separate from `main`:

```text
MLDLOps-Exam2026
```

The commit responsibility is assigned only to:

```text
SK-038 <b22bb038@iitj.ac.in>
```
