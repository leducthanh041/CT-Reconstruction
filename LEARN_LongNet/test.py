import pytorch_lightning as pl

from datamodule import CTDataModule
from models import LEARN_pl


NUM_VIEW = 128
INPUT_SIZE = 256
POISSION_LEVEL = 5e5
PATH_DIR = "/mmlab_students/storageStudents/nguyenvd/Thanhld/CT-Reconstruction/split/"
BATCH_SIZE = 16

CHECKPOINT_PATH = (
    "/mmlab_students/storageStudents/nguyenvd/Thanhld/CT-Reconstruction/LEARN_LongNet/saved_results_noise_2_with_LongNet/results_LEARN_14_iters_bs_1_view_128_noise_500000.0_transform/epoch=24-val_psnr=45.5138.ckpt"
)

SETTING = "numview_" + str(NUM_VIEW) + "_inputsize_256_noise_0"


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
        devices=[3],
        max_epochs=10,
        enable_checkpointing=True,
    )


model = load_model()
dm = build_datamodule()
trainer = build_trainer()

trainer.test(model, dataloaders=dm)