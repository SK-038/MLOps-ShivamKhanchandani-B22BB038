import time
import torch
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import CLIPTokenizer
from diffusers import PNDMScheduler

ONNX_DIR = "./onnx_models"
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.5
NUM_RUNS = 5

# Given prompts to test
PROMPTS = [
    "Bill Gates with a hoodie",
    "John Oliver with Naruto style",
    "Hello Kitty with Naruto style",
    "Lebron James with a hat",
    "A photograph of an orange cat with Naruto style",
]

tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
scheduler = PNDMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

text_encoder_sess = ort.InferenceSession(f"{ONNX_DIR}/text_encoder.onnx", sess_options)
unet_sess = ort.InferenceSession(f"{ONNX_DIR}/unet.onnx", sess_options)
vae_sess = ort.InferenceSession(f"{ONNX_DIR}/vae_decoder.onnx", sess_options)

def encode_text(prompt):
    tokens = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="np",
    )
    text_embeddings = text_encoder_sess.run(None, {"input_ids": tokens.input_ids.astype(np.int64)})[0]
    uncond_tokens = tokenizer(
        [""],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="np",
    )
    uncond_embeddings = text_encoder_sess.run(None, {"input_ids": uncond_tokens.input_ids.astype(np.int64)})[0]
    return np.concatenate([uncond_embeddings, text_embeddings])

def generate_image(prompt, seed=42):
    rng = np.random.RandomState(seed)
    text_embeddings = encode_text(prompt)

    scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    latents = rng.randn(1, 4, 64, 64).astype(np.float32) * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        latent_input = np.concatenate([latents] * 2)
        noise_pred = unet_sess.run(
            None,
            {
                "sample": latent_input,
                "timestep": np.array([t], dtype=np.int64),
                "encoder_hidden_states": text_embeddings,
            },
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred[0:1], noise_pred[1:2]
        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(
            torch_tensor_from_np(noise_pred), t, torch_tensor_from_np(latents)
        ).prev_sample.numpy()

    latents = latents / 0.18215
    image = vae_sess.run(None, {"latent_sample": latents.astype(np.float32)})[0]
    image = (image / 2 + 0.5).clip(0, 1)
    image = (image[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(image)

def torch_tensor_from_np(arr):
    return torch.from_numpy(arr)

if __name__ == "__main__":
    latencies = []

    for i, prompt in enumerate(PROMPTS):
        print(f"\nPrompt {i+1}: {prompt}")
        run_times = []
        for run in range(NUM_RUNS):
            start = time.time()
            img = generate_image(prompt, seed=run)
            elapsed = time.time() - start
            run_times.append(elapsed)
            print(f"  Run {run+1}: {elapsed:.2f}s")
            img.save(f"output_prompt{i+1}_run{run+1}.png")

        avg = sum(run_times) / NUM_RUNS
        latencies.append(avg)
        print(f"  Average: {avg:.2f}s")

    overall_avg = sum(latencies) / len(latencies)
    print(f"\n{'='*50}")
    print(f"Average inference latency (Python ONNX): {overall_avg:.2f} seconds")
    print(f"{'='*50}")