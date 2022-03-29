from typing import Any, Dict, List, Generic, Optional, TypeVar
T = TypeVar("T")  # the output type of `run`

from pytorch_lightning.loops import FitLoop
from torch.utils.data import DataLoader
import numpy as np

def concat_batches(batches: List[Dict]):
    result = {}
    for batch in batches:
        for k,v in batch:
            if k in result:
                result[k] = np.concatenate([result[k],v])
            else:
                result[k] = v


class NonNeuralLoop(FitLoop):
    """ This loop is used to train non-neural models. We Iterate over the dataloader, save 
        the results, and pass them forward to the model without calculating gradients 
        or performing an optimization step.
    """
    
    def advance(self, all_data: Dict, *args: Any, **kwargs: Any) -> None:
        self.trainer.lightning_module.training_step(all_data, 0)
    
    def run(self, dataloader: DataLoader) -> T:
        all_batches =[]
        for batch in dataloader:
            all_batches.append(batch)
        
        all_data = concat_batches(all_batches)
        self.advance(all_data)

