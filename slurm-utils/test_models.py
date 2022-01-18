import pytorch_lightning as pl
from petastorm import make_reader
from petastorm.pytorch import DataLoader as PetastormDataLoader

from src.models.tasks import get_task_from_config_path
from src.models.models import CNNToTransformerEncoder

TEST_PATH = "/gscratch/bdata/mikeam/SeattleFluStudy/data/processed/split/audere_split_2020_02_10_to_2020_04_01/test_7_day_no_scale/"

paths = [
         ("kdd2022/2zot4znv/checkpoints/epoch=49-.ckpt", "./src/data/task_configs/PredictCough.yaml"),
         ("kdd2022/34aa8vp7/checkpoints/epoch=49-.ckpt", "./src/data/task_configs/PredictMobilityDifficulty.yaml"),
         ("kdd2022/2a16o8fi/checkpoints/epoch=49-.ckpt", "./src/data/task_configs/PredictSevereSymptoms.yaml"),
         ("kdd2022/1u1z14s3/checkpoints/epoch=49-.ckpt", "./src/data/task_configs/PredictFluSymptoms.yaml"),
         ("kdd2022/1u8uk751/checkpoints/epoch=49-.ckpt", "./src/data/task_configs/PredictFluPos.yaml"),
         #("kdd2022/2hbzo3mu/checkpoints/epoch=49-.ckpt", "./src/data/task_configs/PredictFatigue.yaml"),
         ]  

for model_path, task_path in paths:
    trainer = pl.Trainer(gpus = -1,
                         accelerator="ddp",
                         resume_from_checkpoint = model_path)
    model = CNNToTransformerEncoder.load_from_checkpoint(model_path)
    task = get_task_from_config_path(task_path)

    with PetastormDataLoader(make_reader(task.test_url,transform_spec=task.transform),
                                   batch_size=300) as test_dataset:
            trainer.test(test_dataloaders=test_dataset)
