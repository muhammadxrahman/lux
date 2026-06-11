from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INFER_", protected_namespaces=()
    )

    model_path: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    host: str = "127.0.0.1"
    port: int = 8000
    max_tokens: int = 512
    
    max_batch_size: int = 8


settings = Settings()