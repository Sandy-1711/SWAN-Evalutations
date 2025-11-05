from llama_cpp import Llama
from app.config import settings
from app.llm_models.simulated_llms import (
    coder_llm,
    compressor_llm,
    generator_llm,
    base_llm as simulated_base_llm,
    baseline_llm as simulated_baseline_llm,
)

if settings.environment == "local":
    coder_llm_2048 = coder_llm
    compressor_llm_2048 = compressor_llm
    generator_llm_2048 = generator_llm
    base_llm = simulated_base_llm
    baseline_llm = simulated_baseline_llm

else:
    coder_llm_2048 = Llama(
        model_path="model_files/coder_F32/unsloth.F32.gguf",
        n_ctx=2048,
        verbose=False,
        use_mmap=True,
        use_mlock=True,
        n_gpu_layers=0,  # set >0 if using GPU
    )

    compressor_llm_2048 = Llama(
        model_path="model_files/compressor_F32/unsloth.F32.gguf",
        n_ctx=2048,
        verbose=False,
        use_mmap=True,
        use_mlock=True,
        n_gpu_layers=0,
    )

    generator_llm_2048 = Llama(
        model_path="model_files/generator_F32/unsloth.F32.gguf",
        n_ctx=2048,
        verbose=False,
        use_mmap=True,
        use_mlock=True,
        n_gpu_layers=0,
    )

    baseline_llm = Llama(
        model_path="model_files/baseline_F32/unsloth.F32.gguf",
        n_ctx=2048,
        verbose=False,
        use_mmap=True,
        use_mlock=True,
        n_gpu_layers=0,
    )

    base_llm = Llama(
        model_path="model_files/qwen/unsloth.BF16.gguf",
        n_ctx=2048,
        verbose=False,
        use_mmap=True,
        use_mlock=True,
        n_gpu_layers=0,
    )
