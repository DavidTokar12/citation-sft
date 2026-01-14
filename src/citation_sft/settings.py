from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.train", env_prefix="SFT_")

    data_dir: str
    max_augment: int
    model_name: str

    # Training
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_length: int

    # Logging & saving
    logging_steps: int
    eval_strategy: str
    save_strategy: str
    save_total_limit: int

    # Dataset split
    test_questions: int
    seed: int


settings = Settings()
