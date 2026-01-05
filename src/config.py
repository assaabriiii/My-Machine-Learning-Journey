from dataclasses import dataclass
import yaml


@dataclass
class ExperimentConfig:
    name: str
    seed: int


@dataclass
class DataConfig:
    dataset: str
    implicit: bool
    negatives: int


@dataclass
class ModelConfig:
    type: str
    embedding_dim: int


@dataclass
class TrainConfig:
    epochs: int
    lr: float
    batch_size: int
    weight_decay: float


@dataclass
class EvalConfig:
    k: list
    metrics: list


@dataclass
class Config:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig


def load_config(path: str) -> Config:
    raw = yaml.safe_load(open(path))

    return Config(
        experiment=ExperimentConfig(**raw["experiment"]),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
        eval=EvalConfig(**raw["eval"]),
    )
