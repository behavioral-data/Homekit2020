from typing import Any, Dict, List, Generic, Optional, TypeVar
T = TypeVar("T")  # the output type of `run`

from pytorch_lightning.loops import FitLoop
from pytorch_lightning.loops.optimization import OptimizerLoop

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


class DummyOptimizerLoop(OptimizerLoop):
    def __init__(self) -> None:
        super().__init__()
    
    def advance(self, _batch: Any, *_args: Any, **_kwargs: Any) -> None:
        return None
    

class NonNeuralLoop(FitLoop):
    """ This loop is used to train non-neural models. We Iterate over the dataloader, save 
        the results, and pass them forward to the model without calculating gradients 
        or performing an optimization step.

        Some ideas on how this should work:
        https://github.com/PyTorchLightning/pytorch-lightning/issues/559
    """
    
    def __init__(self, min_epochs: Optional[int] = 1, max_epochs: int = 1000) -> None:
        super().__init__(min_epochs, max_epochs)
        

    def advance(self, all_data: Dict, *args: Any, **kwargs: Any) -> None:
        self.trainer.lightning_module.training_step(all_data, 0)
    
    def run(self, dataloader: DataLoader) -> T:
        all_batches =[]
        for batch in dataloader:
            all_batches.append(batch)
        
        all_data = concat_batches(all_batches)
        self.advance(all_data)

