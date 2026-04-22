import os
import torch
import shutil
from diffusers import StableDiffusionPipeline
from peft import PeftModel

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
LORA_DIR = "./lora_weights"
ONNX_DIR = "./onnx_models"
os.makedirs(ONNX_DIR, exist_ok=True)

print("Loading base pipeline...")
pipeline = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32)

print("Merging LoRA weights into UNet...")
pipeline.unet = PeftModel.from_pretrained(pipeline.unet, LORA_DIR)
pipeline.unet = pipeline.unet.merge_and_unload()
print("LoRA merged successfully.")

device = "cpu"
pipeline = pipeline.to(device)

# Export Text Encoder
print("\nExporting Text Encoder to ONNX...")
text_encoder = pipeline.text_encoder
text_encoder.eval()
dummy_input_ids = torch.zeros((1, pipeline.tokenizer.model_max_length), dtype=torch.long)

torch.onnx.export(
    text_encoder,
    dummy_input_ids,
    os.path.join(ONNX_DIR, "text_encoder.onnx"),
    input_names=["input_ids"],
    output_names=["last_hidden_state", "pooler_output"],
    dynamic_axes={"input_ids": {0: "batch_size"}},
    opset_version=14,
)
print("Text Encoder exported.")

# Export UNet
print("\nExporting UNet to ONNX...")
unet = pipeline.unet
unet.eval()
dummy_latents = torch.randn(1, 4, 64, 64)
dummy_timestep = torch.tensor([1])
dummy_encoder_hidden = torch.randn(1, 77, 768)

torch.onnx.export(
    unet,
    (dummy_latents, dummy_timestep, dummy_encoder_hidden),
    os.path.join(ONNX_DIR, "unet.onnx"),
    input_names=["sample", "timestep", "encoder_hidden_states"],
    output_names=["out_sample"],
    dynamic_axes={
        "sample": {0: "batch_size"},
        "encoder_hidden_states": {0: "batch_size"},
    },
    opset_version=14,
)
print("UNet exported.")

# Export VAE Decoder
print("\nExporting VAE Decoder to ONNX...")
class VAEDecoder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        return self.vae.decode(latents).sample

vae_decoder = VAEDecoder(pipeline.vae)
vae_decoder.eval()
dummy_vae_input = torch.randn(1, 4, 64, 64)

torch.onnx.export(
    vae_decoder,
    dummy_vae_input,
    os.path.join(ONNX_DIR, "vae_decoder.onnx"),
    input_names=["latent_sample"],
    output_names=["sample"],
    dynamic_axes={"latent_sample": {0: "batch_size"}},
    opset_version=14,
)
print("VAE Decoder exported.")

# Size Report
def dir_size_gb(path):
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(path)
        for f in files
    )
    return total / (1024 ** 3)

onnx_size = dir_size_gb(ONNX_DIR)
print(f"\nONNX models saved to: {ONNX_DIR}")
print(f"Combined ONNX model size: {onnx_size:.2f} GB")
print("(Base model size: check ~/.cache/huggingface/hub for exact size, typically ~4 GB)")