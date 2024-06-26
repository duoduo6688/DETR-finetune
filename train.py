import pytorch_lightning as pl
import torch
from detr_model import Detr
from dataset import DataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from model_type_map import MODEL_TYPE_MAP
import wandb

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


def main(wandb_logger, batch_size, experiment_name, lr, lr_backbone, weight_decay, num_queries, lr_decay_steps, **kwargs):
    sku_data_module = DataModule(batch_size=batch_size, dataset_name="SKU110K", model_type=MODEL_TYPE_MAP[kwargs["model_type"]])
    output_folder = f"../checkpoints/{experiment_name}"
    callbacks = [
            ModelCheckpoint(monitor='map',
                            dirpath=output_folder + "/",
                            filename="{epoch}-{step}-{validation_loss:.3f}-{map:.3f}",
                            mode="max",
                            every_n_epochs=1,
                            save_top_k=3,
			                save_last=True,
                            ),
        ]
    model = Detr(lr=lr, lr_backbone=lr_backbone, weight_decay=weight_decay, num_queries=num_queries, lr_decay_steps=lr_decay_steps, **kwargs)
    #TODO there should be a better way to do this
    if PARAMS["accumulate_grad_batches"]:
        trainer = pl.Trainer(
            accelerator="gpu",
            max_epochs=300,
            precision='16-mixed',
            benchmark=True,
            callbacks=callbacks,
            logger=wandb_logger,
            accumulate_grad_batches=PARAMS["accumulate_grad_batchesches"],
        )
    else:
        trainer = pl.Trainer(
            accelerator="gpu",
            max_epochs=300,
            precision='16-mixed',
            benchmark=True,
            callbacks=callbacks,
            logger=wandb_logger,
        )
    ckpt_path = f"{output_folder}/last.ckpt"
    trainer.fit(model,
                sku_data_module,
                ckpt_path=ckpt_path,
                )
    print("trainer fit done")
    torch.save(model.state_dict(), f"{output_folder}/model.pt")
    print("save model.pt success")
    torch.save(model, f"{output_folder}/pytorch_model.bin")
    print("save pytorch_model.bin success")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    wandb_logger = WandbLogger(project="DETR-finetune-num-queries-400-sku110k", save_dir="..")
    wandb_logger.log_hyperparams(PARAMS)
    code_artifact = wandb.Artifact('codebase', type='code')
    code_artifact.add_dir('.')
    wandb.log_artifact(code_artifact)
    main(wandb_logger, **PARAMS)
