from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INFER_", protected_namespaces=()
    )

    model_path: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    host: str = "127.0.0.1"
    port: int = 8000
    max_tokens: int = 512
    
    # Continuous batching. The scheduler runs one long-lived BatchGenerator;
    # these cap how many sequences decode and prefill concurrently. They map to
    # BatchGenerator's completion_batch_size / prefill_batch_size.
    max_concurrent_seqs: int = 16
    prefill_batch_size: int = 8
    system_prompt: str = (
        "You are a helpful assistant. Answer concisely and accurately. "
        "Always be polite and clear, and never make up information."
    )
    enable_prefix_cache: bool = True


settings = Settings()