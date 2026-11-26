import torch
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from workload.dataset import WorkloadInput, WorkloadMeasurements


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass(frozen=True)
class PreprocessingConfig:
    task_type_mapping: dict[str, int]
    token_count_mean: float
    token_count_std: float
    complexity_mean: float
    complexity_std: float
    target_mean: torch.Tensor
    target_std: torch.Tensor


def embed_prompts(prompts: list[str]) -> torch.Tensor:
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        prompts,
        convert_to_tensor=True,
        show_progress_bar=False,
    )

    return embeddings.to(dtype=torch.float32)


def fit_preprocessing_config(
    examples: list[tuple[WorkloadInput, WorkloadMeasurements]],
) -> PreprocessingConfig:
    if not examples:
        raise ValueError("Cannot fit preprocessing configuration on empty data.")

    inputs = [example[0] for example in examples]
    measurements = [example[1] for example in examples]

    if any(item.input_token_count is None for item in inputs):
        raise ValueError("Input token counts must be available.")

    task_types = sorted({item.task_type for item in inputs})

    task_type_mapping = {
        task_type: index
        for index, task_type in enumerate(task_types)
    }

    complexity_scores = torch.tensor(
        [item.complexity for item in inputs],
        dtype=torch.float32,
    )

    complexity_mean = complexity_scores.mean().item()
    complexity_std = max(
        complexity_scores.std(correction=0).item(),
        1e-8,
    )

    input_token_counts = torch.tensor(
        [item.input_token_count for item in inputs],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [
            [
                item.output_token_count,
                item.runtime_seconds,
                item.energy_joules,
            ]
            for item in measurements
        ],
        dtype=torch.float32,
    )

    token_count_mean = input_token_counts.mean().item()
    token_count_std = max(
        input_token_counts.std(correction=0).item(),
        1e-8,
    )

    target_mean = targets.mean(dim=0)
    target_std = targets.std(dim=0, correction=0).clamp(min=1e-8)

    return PreprocessingConfig(
        task_type_mapping=task_type_mapping,
        token_count_mean=token_count_mean,
        token_count_std=token_count_std,
        complexity_mean=complexity_mean,
        complexity_std=complexity_std,
        target_mean=target_mean,
        target_std=target_std,
    )


def transform_inputs(
    inputs: list[WorkloadInput],
    config: PreprocessingConfig,
) -> torch.Tensor:
    if not inputs:
        raise ValueError("Cannot transform empty input data.")

    if any(item.input_token_count is None for item in inputs):
        raise ValueError("Input token counts must be available.")

    prompts = [item.prompt for item in inputs]
    embeddings = embed_prompts(prompts).cpu()

    input_token_counts = torch.tensor(
        [[item.input_token_count] for item in inputs],
        dtype=torch.float32,
    )

    input_token_counts = (
        input_token_counts - config.token_count_mean
    ) / config.token_count_std

    task_types = torch.zeros(
        len(inputs),
        len(config.task_type_mapping),
        dtype=torch.float32,
    )


    for index, item in enumerate(inputs):
        task_type_index = config.task_type_mapping[item.task_type]

        task_types[index, task_type_index] = 1.0

        complexity_scores = torch.tensor(
            [[item.complexity] for item in inputs],
            dtype=torch.float32,
        )

        complexity_scores = (
            complexity_scores - config.complexity_mean
        ) / config.complexity_std

    return torch.cat(
        [
            embeddings,
            input_token_counts,
            task_types,
            complexity_scores,
        ],
        dim=1,
    )