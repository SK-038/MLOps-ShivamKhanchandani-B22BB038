#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <random>
#include <cmath>
#include <string>
#include <onnxruntime_cxx_api.h>

// Minimal pipeline: UNet denoising loop only (latents pre-encoded, embeddings pre-computed)
// For full pipeline, text encoder and VAE would be added the same way.

static const int   NUM_RUNS            = 5;
static const int   NUM_INFERENCE_STEPS = 20;
static const float GUIDANCE_SCALE      = 7.5f;
static const int   LATENT_H            = 64;
static const int   LATENT_W            = 64;
static const int   LATENT_C            = 4;
static const int   EMBED_DIM           = 768;
static const int   SEQ_LEN             = 77;

std::vector<float> randn_vector(size_t n, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(gen);
    return v;
}

double run_unet_benchmark(const std::string& unet_path) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sd_cpp");
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session unet_session(env, unet_path.c_str(), opts);
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<double> latencies;

    for (int run = 0; run < NUM_RUNS; ++run) {
        auto start = std::chrono::high_resolution_clock::now();

        size_t latent_size = LATENT_C * LATENT_H * LATENT_W;
        auto latents = randn_vector(latent_size, run);

        // Dummy text embeddings (2 x SEQ_LEN x EMBED_DIM — uncond + cond)
        std::vector<float> embeddings(2 * SEQ_LEN * EMBED_DIM, 0.f);

        // Simple linear noise schedule
        for (int step = NUM_INFERENCE_STEPS - 1; step >= 0; --step) {
            int64_t timestep_val = step;

            // Duplicate latents for classifier-free guidance
            std::vector<float> latent_input(latents.begin(), latents.end());
            latent_input.insert(latent_input.end(), latents.begin(), latents.end());

            std::vector<int64_t> latent_shape  = {2, LATENT_C, LATENT_H, LATENT_W};
            std::vector<int64_t> t_shape        = {1};
            std::vector<int64_t> embed_shape    = {2, SEQ_LEN, EMBED_DIM};

            auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value latent_tensor = Ort::Value::CreateTensor<float>(
                mem_info, latent_input.data(), latent_input.size(),
                latent_shape.data(), latent_shape.size());

            Ort::Value t_tensor = Ort::Value::CreateTensor<int64_t>(
                mem_info, &timestep_val, 1, t_shape.data(), t_shape.size());

            Ort::Value embed_tensor = Ort::Value::CreateTensor<float>(
                mem_info, embeddings.data(), embeddings.size(),
                embed_shape.data(), embed_shape.size());

            const char* input_names[]  = {"sample", "timestep", "encoder_hidden_states"};
            const char* output_names[] = {"out_sample"};

            std::vector<Ort::Value> inputs;
            inputs.push_back(std::move(latent_tensor));
            inputs.push_back(std::move(t_tensor));
            inputs.push_back(std::move(embed_tensor));

            auto outputs = unet_session.Run(
                Ort::RunOptions{nullptr},
                input_names, inputs.data(), 3,
                output_names, 1);

            float* noise_pred = outputs[0].GetTensorMutableData<float>();

            // Classifier-free guidance + simple DDPM step
            float alpha = 1.f - (float)step / NUM_INFERENCE_STEPS;
            for (size_t i = 0; i < latent_size; ++i) {
                float uncond = noise_pred[i];
                float cond   = noise_pred[latent_size + i];
                float guided = uncond + GUIDANCE_SCALE * (cond - uncond);
                latents[i]   = latents[i] - alpha * guided;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        latencies.push_back(elapsed);
        std::cout << "Run " << (run + 1) << ": " << elapsed << "s\n";
    }

    double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / NUM_RUNS;
    return avg;
}

int main(int argc, char* argv[]) {
    std::string unet_path = (argc > 1) ? argv[1] : "./onnx_models/unet.onnx";

    std::cout << "=== C++ ONNX Runtime Inference Benchmark ===\n";
    std::cout << "UNet model: " << unet_path << "\n";
    std::cout << "Runs: " << NUM_RUNS << "\n\n";

    double avg = run_unet_benchmark(unet_path);

    std::cout << "\n==================================================\n";
    std::cout << "Average inference latency (C++ ONNX): " << avg << " seconds\n";
    std::cout << "==================================================\n";
    return 0;
}