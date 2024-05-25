
from detr_model import Detr
import torch

PARAMS = {
        "lr": 1e-4,
        "lr_backbone": 1e-5,
        "weight_decay": 1e-4,
        "num_queries": 400,
        "lr_decay_steps": 70,
        "batch_size": 8,
        "train_backbone": False,
        "experiment_name": "detr_train_v4_cont_backbone",
        "model_type": "resnet-50",
        "augmentations": ["hflip", "blur"],
        "accumulate_grad_batches": None
    }

def main(batch_size, experiment_name, lr, lr_backbone, weight_decay, num_queries, lr_decay_steps, **kwargs):
    output_folder = f"../checkpoints/{experiment_name}"
    ckpt_path = f"{output_folder}/last.ckpt"
    model = Detr(lr=lr, lr_backbone=lr_backbone, weight_decay=weight_decay, num_queries=num_queries, lr_decay_steps=lr_decay_steps, **kwargs)
    model.load_from_checkpoint(ckpt_path)
    torch.save(model, "pytorch_model.bin")

if __name__ == "__main__":
    main(**PARAMS)