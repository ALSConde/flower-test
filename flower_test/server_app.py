import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx
import numpy as np
import json
import os
from datetime import datetime
from flower_test.task import Net

# Create ServerApp
app = ServerApp()


def make_serializable(obj):
    """Recursively convert objects to JSON-serializable types."""
    # Basic simple types
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    # Dict -> convert values
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    # List/tuple -> list
    if isinstance(obj, (list, tuple, set)):
        return [make_serializable(v) for v in obj]
    # Numpy
    try:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    # Torch tensors -> lists
    try:
        import torch as _torch

        if isinstance(obj, _torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    # Fallback to string representation
    try:
        return str(obj)
    except Exception:
        return None


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    test_name: str = context.run_config["test-name"]
    proximal_mu: float = context.run_config["proximal-mu"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    if test_name == "fedavg":
        strategy = FedAvg(fraction_train=fraction_train)
    elif test_name == "fedprox":
        strategy = FedProx(fraction_train=fraction_train, proximal_mu=proximal_mu)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, f"{test_name}_model.pt")

    # Compose the results dictionary
    results = {
        "run_config": {
            "fraction_train": fraction_train,
            "num_server_rounds": num_rounds,
            "lr": lr,
        },
        # small summary of model: parameter shapes
        "model_param_shapes": {
            k: list(v.size()) if hasattr(v, "size") else None
            for k, v in state_dict.items()
        },
        "result_summary": {},
    }

    for attr in ("metrics", "loss", "num_rounds", "num_examples"):
        if hasattr(result, attr):
            try:
                results["result_summary"][attr] = make_serializable(
                    getattr(result, attr)
                )
            except Exception:
                results["result_summary"][attr] = str(getattr(result, attr))

    if hasattr(result, "__dict__"):
        for k, v in vars(result).items():
            if k in results["result_summary"]:
                continue
            try:
                results["result_summary"][k] = make_serializable(v)
            except Exception:
                results["result_summary"][k] = str(v)

    # Write JSON to disk
    try:
        test_name = test_name+f"_my={proximal_mu}" if test_name == "fedprox" else test_name
        with open(f"train_results_{test_name}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Training results saved to train_results_{test_name}.json")
    except Exception as e:
        print(f"Warning: failed to write train_results_{test_name}.json: {e}")
