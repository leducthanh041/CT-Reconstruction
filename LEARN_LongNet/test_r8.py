import pytorch_lightning as pl

from datamodule import CTDataModule
from models_r8 import LEARN_pl


NUM_VIEW = 64
INPUT_SIZE = 256
POISSION_LEVEL = 1e6
PATH_DIR = "/mmlab_students/storageStudents/nguyenvd/Thanhld/CT-Reconstruction/split/"
BATCH_SIZE = 16

CHECKPOINT_PATH = (
    "/mmlab_students/storageStudents/nguyenvd/Thanhld/CT-Reconstruction/LEARN_LongNet/saved_results_noise_2_with_r8/results_LEARN_14_iters_bs_1_view_64_noise_1000000.0_transform/epoch=00-val_psnr=31.0005.ckpt"
)

SETTING = "numview_" + str(NUM_VIEW) + "_inputsize_256_noise_0_transform"


def load_model():
    return LEARN_pl.load_from_checkpoint(CHECKPOINT_PATH)


def build_datamodule():
    datamodule = CTDataModule(
        data_dir=PATH_DIR,
        batch_size=BATCH_SIZE,
        num_view=NUM_VIEW,
        input_size=INPUT_SIZE,
        setting=SETTING,
        poission_level=POISSION_LEVEL,
    )
    datamodule.setup("test")
    return datamodule


def build_trainer():
    return pl.Trainer(
        accelerator="gpu",
        devices=[2],
        max_epochs=10,
        enable_checkpointing=True,
    )


model = load_model()
dm = build_datamodule()
trainer = build_trainer()

trainer.test(model, dataloaders=dm)