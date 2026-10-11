import os
import json
import time
import torch
import requests
from PIL import Image
import io
from transformers import AutoModel, AutoProcessor

# Configuration
# The claim specifies 'EmbeddingGemma 2' is 740M. 
# We will attempt to load the model that matches the claim's description.
# Note: In a real environment, the exact ID would be used. 
# For this experiment, we target the model ID that would represent the 740M version.
MODEL_ID = "google/embedding-gemma-2-9b" 
CLAIM_PARAM_COUNT = 740_000_000
SEED = 42
torch.manual_seed(SEED)

def get_model_params(model):
    return sum(p.numel() for p in model.parameters())

def write_results(results):
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

def run_experiment():
    results = {
        "metrics": {},
        "observations": [],
        "passed": None
    }
    
    start_time = time.perf_counter()
    
    # 1. Verify Parameter Count
    try:
        print(f"Attempting to load model: {MODEL_ID}")
        # Using device='cpu' as per environment constraints
        model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, device_map="cpu")
        model.eval()
        
        actual_params = get_model_params(model)
        results["metrics"]["actual_parameter_count"] = actual_params
        results["metrics"]["claimed_parameter_count"] = CLAIM_PARAM_COUNT
        
        param_diff_ratio = abs(actual_params - CLAIM_PARAM_COUNT) / CLAIM_PARAM_COUNT
        results["observations"].append(f"Parameter count: {actual_params} (Claimed: {CLAIM_PARAM_COUNT})")
        
        # 5% tolerance for parameter count
        param_passed = param_diff_ratio <= 0.05
        if not param_passed:
            results["observations"].append("Parameter count deviates significantly from claim.")
        else:
            results["observations"].append("Parameter count matches claim within tolerance.")

    except Exception as e:
        print(f"Error loading model: {e}")
        results["observations"].append(f"Failed to load model: {str(e)}")
        results["passed"] = False
        write_results(results)
        return

    # 2. Verify Multimodal Dimensionality
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        
        # Test Inputs
        text_input = "A beautiful sunset over the mountains."
        img = Image.new('RGB', (224, 224), color = (73, 109, 137))
        
        # Text Embedding
        with torch.no_grad():
            text_inputs = processor(text=text_input, return_tensors="pt")
            text_outputs = model(**text_inputs)
            # Handle different possible output structures for embedding models
            if hasattr(text_outputs, 'embeddings'):
                text_emb = text_outputs.embeddings
            elif hasattr(text_outputs, 'last_hidden_state'):
                text_emb = text_outputs.last_hidden_state
            else:
                text_emb = text_outputs[0]
            
            text_dim = text_emb.shape[-1]

        # Image Embedding
        with torch.no_grad():
            image_inputs = processor(images=img, return_tensors="pt")
            image_outputs = model(**image_inputs)
            if hasattr(image_outputs, 'embeddings'):
                image_emb = image_outputs.embeddings
            elif hasattr(image_outputs, 'last_hidden_state'):
                image_emb = image_outputs.last_hidden_state
            else:
                image_emb = image_outputs[0]
                
            image_dim = image_emb.shape[-1]

        results["metrics"]["text_embedding_dim"] = text_dim
        results["metrics"]["image_embedding_dim"] = image_dim
        results["observations"].append(f"Text dim: {text_dim}, Image dim: {image_dim}")

        dim_passed = (text_dim == image_dim)
        if not dim_passed:
            results["observations"].append("Modalities produce different vector dimensions.")
        else:
            results["observations"].append("Modalities produce identical vector dimensions.")

        # Final Verdict
        results["passed"] = param_passed and dim_passed

    except Exception as e:
        print(f"Error during inference: {e}")
        results["observations"].append(f"Inference error: {str(e)}")
        results["passed"] = None

    results["execution_time_sec"] = time.perf_counter() - start_time
    write_results(results)

def main():
    run_experiment()
    
    if os.path.exists("results.json"):
        with open("results.json", "r") as f:
            res = json.load(f)
        
        print("\n--- Experiment Summary ---")
        print(f"Passed: {res['passed']}")
        print("Observations:")
        for obs in res['observations']:
            print(f" - {obs}")
        print("--------------------------")
    else:
        print("Failed to generate results.json")

if __name__ == "__main__":
    main()