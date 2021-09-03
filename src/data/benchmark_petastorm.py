import torch
from petastorm.pytorch import DataLoader
import numpy as np
import pytorch_lightning as pl
import pandas as pd

from petastorm import make_reader
from petastorm.transform import TransformSpec
from petastorm.etl.dataset_metadata import infer_or_load_unischema
from pyarrow.parquet import ParquetDataset

from src.models.models import CNNToTransformerEncoder

torch.manual_seed(1)



TRAIN_PATH = "/homes/gws/mikeam/seattleflustudy/data/processed/petastorm_datasets/minute_level"
schema = infer_or_load_unischema(ParquetDataset(TRAIN_PATH,validate_schema=False))
fields = [k for k in schema.fields.keys()]
# features = [k for k in schema.fields.keys() if not k in ["start","end","participant_id"]]

def _transform_row(row):
    # labler = self.get_labler()
    start = pd.to_datetime(row.pop("start"))
    #Because spark windows have and exclusive right boundary:
    end = pd.to_datetime(row.pop("end")) - pd.to_timedelta("1ms")

    participant_id = row.pop("participant_id")
    
    keys = sorted(row.keys())

    result = np.vstack([row[k].T for k in keys]).T
    return {"inputs_embeds":result,
            "label": np.random.randint(2)}

new_fields = [("inputs_embeds",np.float32,None,False),
                ("label",np.int_,None,False)]

transform = TransformSpec(_transform_row,removed_fields=fields,
                                            edit_fields= new_fields)

trainer = pl.Trainer(gpus = -1,
                     accelerator="ddp",
                     num_sanity_val_steps=0)
                     

with DataLoader(make_reader('file://'+TRAIN_PATH, num_epochs=1,
                            transform_spec=transform), batch_size=500) as train_loader:
    model = CNNToTransformerEncoder(8,1,1,24*60*4,loss_fn="FocalLoss")
    trainer.fit(model,train_loader,train_loader)

