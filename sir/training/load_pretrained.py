"""
Utilities for loading pre-trained LeRobot policies from various sources.

This module makes it easy to:
1. Load BC (Behavioral Cloning) policies trained with train_act.py
2. Initialize DAgger training with these pre-trained policies
3. Load policies from local checkpoints or W&B artifacts
4. Maintain compatibility with LeRobot's policy loading

Designed specifically for the BC -> DAgger -> RL training pipeline.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import wandb

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_pre_post_processors
from sir.training.wandb_artifacts import download_checkpoint_from_wandb

logger = logging.getLogger(__name__)


def load_policy_from_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
    strict: bool = False,
) -> Tuple[ACTPolicy, dict, dict]:
    """
    Load a complete ACT policy from a local checkpoint directory.

    This loads all necessary components:
    - Policy weights (model.safetensors)
    - Policy configuration (config.json)
    - Preprocessor with normalization statistics
    - Postprocessor with denormalization statistics

    Args:
        checkpoint_path: Path to the checkpoint directory (containing config.json, model.safetensors, etc.)
        device: Device to load model to ('cpu', 'cuda', 'mps', etc.)
        strict: Whether to strictly require all model parameters

    Returns:
        Tuple of (policy, preprocessor, postprocessor)
        - policy: ACTPolicy instance ready for inference or DAgger fine-tuning
        - preprocessor: PolicyProcessorPipeline for input normalization
        - postprocessor: PolicyProcessorPipeline for output denormalization

    Example:
        >>> # Load best policy from local training
        >>> policy, prep, post = load_policy_from_checkpoint(
        ...     "checkpoints/lift_task/best_model",
        ...     device="cuda"
        ... )
        >>> # Use for inference
        >>> obs = {...}
        >>> action = policy.select_action(obs)
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

    logger.info(f"Loading policy from: {checkpoint_path}")

    # Load policy with weights using LeRobot's from_pretrained
    # This automatically handles config loading, including the "type" field
    logger.info("Loading policy and weights...")
    policy = ACTPolicy.from_pretrained(
        str(checkpoint_path),
        strict=strict,
    )
    policy = policy.to(device)
    policy.eval()

    # Load preprocessor and postprocessor with saved statistics
    logger.info("Loading preprocessor and postprocessor...")
    # Override device in processor pipelines to match target device
    # This is important when loading checkpoints trained on different hardware (e.g., CUDA -> MPS)
    device_overrides = {
        "device_processor": {
            "device": device
        }
    }
    logger.info(f"Overriding processor device to: {device}")
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint_path),  # Load saved processors with stats
        preprocessor_overrides=device_overrides,
        postprocessor_overrides=device_overrides,
    )

    logger.info(f"✓ Policy loaded successfully from {checkpoint_path}")

    return policy, preprocessor, postprocessor


def load_policy_from_wandb(
    artifact_identifier: str,
    device: str = "cpu",
    strict: bool = False,
) -> Tuple[ACTPolicy, dict, dict]:
    """
    Load a pre-trained ACT policy from a W&B artifact.

    This is particularly useful for:
    1. Loading BC policies trained on a remote cluster
    2. Sharing models between team members
    3. Starting DAgger training with pre-trained BC policies
    4. Model versioning and reproducibility

    The artifact must contain:
    - config.json (policy configuration)
    - model.safetensors (model weights)
    - policy_preprocessor.json + .safetensors (input normalization)
    - policy_postprocessor.json + .safetensors (output denormalization)

    Args:
        artifact_identifier: W&B artifact identifier in format:
            - "artifact-name:latest" (latest version)
            - "artifact-name:v0" (specific version)
            - "entity/project/artifact-name:version" (full path)
        device: Device to load model to
        strict: Whether to strictly require all model parameters

    Returns:
        Tuple of (policy, preprocessor, postprocessor) - same as load_policy_from_checkpoint

    Example:
        >>> # Load best BC policy from W&B for DAgger
        >>> policy, prep, post = load_policy_from_wandb(
        ...     "self-improving/act-training/act-lift-best:latest",
        ...     device="cuda"
        ... )
        >>> # Now use this as initialization for DAgger training
        >>> # Initialize DAgger trainer with this policy...

    Raises:
        FileNotFoundError: If artifact not found or checkpoint files are missing
    """
    logger.info(f"Downloading artifact from W&B: {artifact_identifier}")

    # Download the artifact (only requires wandb login, not active run)
    checkpoint_path = download_checkpoint_from_wandb(
        artifact_identifier=artifact_identifier,
        download_dir=None,  # Uses default cache directory
    )

    logger.info(f"Artifact downloaded to: {checkpoint_path}")

    # Load the policy from the downloaded checkpoint
    return load_policy_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        strict=strict,
    )


def load_bc_policy_for_dagger(
    checkpoint_source: str | Path,
    device: str = "cuda",
    from_wandb: bool = False,
) -> Tuple[ACTPolicy, dict, dict]:
    """
    Load a BC (Behavioral Cloning) policy to use as initialization for DAgger training.

    This is the recommended function for starting DAgger from BC policies.
    It handles both local and W&B artifact sources transparently.

    Args:
        checkpoint_source: Either:
            - Local path to checkpoint: "checkpoints/lift/best_model"
            - W&B artifact ID: "act-lift-best:latest"
        device: Device to load to (default: 'cuda' for training)
        from_wandb: If True, treat checkpoint_source as W&B artifact ID

    Returns:
        Tuple of (policy, preprocessor, postprocessor)

    Example:
        >>> # Load BC policy trained locally
        >>> policy, prep, post = load_bc_policy_for_dagger(
        ...     "checkpoints/lift_v1/best_model"
        ... )
        >>> # Or load from W&B
        >>> policy, prep, post = load_bc_policy_for_dagger(
        ...     "act-lift-best-step-10000:latest",
        ...     from_wandb=True
        ... )
        >>> # Use for DAgger training...
    """
    if from_wandb:
        logger.info(f"Loading BC policy from W&B: {checkpoint_source}")
        return load_policy_from_wandb(
            artifact_identifier=str(checkpoint_source),
            device=device,
            strict=False,  # Allow loading into different shapes for fine-tuning
        )
    else:
        logger.info(f"Loading BC policy from local checkpoint: {checkpoint_source}")
        return load_policy_from_checkpoint(
            checkpoint_path=checkpoint_source,
            device=device,
            strict=False,
        )


def list_available_bc_policies() -> list[dict]:
    """
    List all available BC policy artifacts in the current W&B project.

    Returns:
        List of dicts with artifact info (name, version, metadata)

    Example:
        >>> policies = list_available_bc_policies()
        >>> for p in policies:
        ...     print(f"{p['name']}: {p['description']}")
    """
    if not wandb.run:
        logger.warning("W&B run not active. Cannot list policies.")
        return []

    api = wandb.Api()
    try:
        artifacts = api.artifacts(
            type_name="policy",
            project=wandb.run.project,
        )
        return [
            {
                "name": a.name,
                "version": a.version,
                "description": a.description,
                "metadata": a.metadata,
                "created_at": a.created_at,
            }
            for a in artifacts
        ]
    except Exception as e:
        logger.error(f"Failed to list policies: {e}")
        return []


# Convenience imports
__all__ = [
    "load_policy_from_checkpoint",
    "load_policy_from_wandb",
    "load_bc_policy_for_dagger",
    "list_available_bc_policies",
]
