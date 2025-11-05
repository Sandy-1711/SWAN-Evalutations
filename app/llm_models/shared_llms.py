from llama_cpp import Llama
from app.config import settings
from app.llm_models.simulated_llms import (
    coder_llm,
    compressor_llm,
    generator_llm,
    base_llm,
    baseline_llm,
)

if settings.environment == "local":
    coder_llm_2048 = coder_llm
    compressor_llm_2048 = compressor_llm
    generator_llm_2048 = generator_llm
    base_llm = base_llm
    baseline_llm = baseline_llm
    pass


coder_llm_2048 = Llama(
    model_path="models/coder_F32/unsloth.F32.gguf",
    n_ctx=2048,
    verbose=False,
    use_mmap=True,
    use_mlock=True,
    # n_threads=16,
    n_gpu_layers=0,
)

compressor_llm_2048 = Llama(
    model_path="models/compressor_F32/unsloth.F32.gguf",
    n_ctx=2048,
    use_mmap=True,
    use_mlock=True,
    n_gpu_layers=0,
    verbose=False,
)
generator_llm_2048 = Llama(
    model_path="models/generator_F32/unsloth.F32.gguf",
    n_ctx=2048,
    verbose=False,
    use_mmap=True,
    use_mlock=True,  # Set True only if you want to lock it in RAM
    n_gpu_layers=0,  # if CPU-only
)

baseline_llm = Llama(
    model_path="models/baseline_F32/unsloth.F32.gguf",
    n_ctx=2048,
    verbose=False,
    use_mmap=True,
    use_mlock=True,  # Set True only if you want to lock it in RAM
    n_gpu_layers=0,  # if CPU-only
)

base_llm = Llama(
    model_path="models/qwen/unsloth.BF16.gguf",
    n_ctx=2048,
    # n_threads=N_THREADS,
    use_mlock=True,
    use_mmap=True,
    n_gpu_layers=0,  # if CPU-only
    verbose=False,
)
