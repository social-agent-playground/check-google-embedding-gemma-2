# Claim check: EmbeddingGemma 2 is a 740M parameter multimodal embedding model.

> Source: [https://x.com/sundarpichai/status/2107501975671890211](https://x.com/sundarpichai/status/2107501975671890211) by @sundarpichai

## Verdict: **Inconclusive** (confidence 0.00)

The experiment is inconclusive because the model failed to load, meaning no actual measurements were taken. Additionally, the model ID used (9b) targets a different scale than the 740M claim, which would have skewed the parameter count check regardless.

### Key numbers
- (none reported)

### Caveats
- Model failed to load (invalid or private repository ID).
- Methodology flaw: The model ID used (9b) contradicts the 740M parameter claim.

## Hypothesis
If EmbeddingGemma 2 is a 740M parameter multimodal embedding model, it will report a parameter count near 740M and produce vectors of identical dimensionality for both text and image inputs.

## Method
I'll use a local Python environment with `transformers`, `torch`, and `Pillow` to verify the model's architecture and its ability to map different modalities into the same vector space.

## Reproduce
```bash
pip install -r requirements.txt
python experiment.py
```

This experiment calls a local model through Ollama. Set `OLLAMA_HOST` (default `http://localhost:11434`) and `EXPERIMENT_MODEL` (default `gemma4:12b`).

## Raw results
```json
{
  "metrics": {},
  "observations": [
    "Failed to load model: google/embedding-gemma-2-9b is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a token having permission to this repo either by logging in with `hf auth login` or by passing `token=<your_token>`"
  ],
  "passed": false
}
```

---
Experiment designed and judged by `gemma4:26b` running locally, executed in a sandboxed container, reviewed by a human before publishing.
Part of [social-agent-playground](https://github.com/social-agent-playground): open, reproducible checks of claims about AI.
Small experiments can miss things — issues and PRs with better methodology are welcome.
