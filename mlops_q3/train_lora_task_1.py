import os
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from huggingface_hub import login, upload_folder
from tqdm import tqdm

MODEL_ID      = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DATASET_ID    = "lambda/naruto-blip-captions"
OUTPUT_DIR    = "./lora_weights"
HF_REPO_ID    = os.environ.get("HF_REPO_ID", "Shivam_K/naruto-lora-sd15")
RESOLUTION    = 512
BATCH_SIZE    = 1
LEARNING_RATE = 1e-4
MAX_STEPS     = 500
LORA_RANK     = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("Logged in to HuggingFace.")
else:
    print("HF_TOKEN not set — skipping HuggingFace login.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

pipeline     = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
unet         = pipeline.unet
tokenizer    = pipeline.tokenizer
text_encoder = pipeline.text_encoder
vae          = pipeline.vae
scheduler    = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

total_base_params = sum(p.numel() for p in unet.parameters())
print(f"Total base UNet parameters: {total_base_params:,}")

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.1,
    bias="none",
)
unet = get_peft_model(unet, lora_config)

trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
total_with_lora  = sum(p.numel() for p in unet.parameters())
print(f"Trainable LoRA parameters:      {trainable_params:,}")
print(f"Total parameters (Base + LoRA): {total_with_lora:,}")

image_transforms = transforms.Compose([
    transforms.Resize((RESOLUTION, RESOLUTION)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def collate_fn(examples):
    pixel_values = torch.stack([
        image_transforms(example["image"].convert("RGB"))
        for example in examples
    ])
    input_ids = tokenizer(
        [example["text"] for example in examples],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids
    return {"pixel_values": pixel_values, "input_ids": input_ids}

dataset    = load_dataset(DATASET_ID, split="train")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

unet         = unet.to(device)
vae          = vae.to(device)
text_encoder = text_encoder.to(device)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.train()

optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)

global_step = 0
final_loss  = 0.0

for epoch in range(9999):
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        if global_step >= MAX_STEPS:
            break

        pixel_values = batch["pixel_values"].to(device)
        input_ids    = batch["input_ids"].to(device)

        with torch.no_grad():
            latents               = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            encoder_hidden_states = text_encoder(input_ids)[0]

        noise         = torch.randn_like(latents)
        timesteps     = torch.randint(
            0, scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=device
        ).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
        loss       = torch.nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_loss   = loss.item()
        global_step += 1

        if global_step % 100 == 0:
            print(f"Step {global_step}/{MAX_STEPS} | Loss: {final_loss:.4f}")

    if global_step >= MAX_STEPS:
        break

unet.save_pretrained(OUTPUT_DIR)

lora_size = sum(
    os.path.getsize(os.path.join(OUTPUT_DIR, f))
    for f in os.listdir(OUTPUT_DIR)
) / (1024 * 1024)

print(f"\nLoRA weights saved to : {OUTPUT_DIR}")
print(f"Final training loss   : {final_loss:.4f}")
print(f"LoRA adapter size     : {lora_size:.2f} MB")

if hf_token:
    from huggingface_hub import create_repo
    create_repo(HF_REPO_ID, exist_ok=True, token=hf_token)
    upload_folder(
        folder_path=OUTPUT_DIR,
        repo_id=HF_REPO_ID,
        token=hf_token,
    )
    print(f"Uploaded to: https://huggingface.co/{HF_REPO_ID}")
else:
    print("Skipping HuggingFace upload — HF_TOKEN not set.")