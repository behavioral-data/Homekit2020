from pytorch_lightning.loggers.wandb import WandbLogger, _WANDB_GREATER_EQUAL_0_12_10, _WANDB_GREATER_EQUAL_0_10_22
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union
from weakref import ReferenceType

from pytorch_lightning.callbacks import Checkpoint
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.loggers.logger import Logger

try:
    import wandb
    from wandb.sdk.lib import RunDisabled
    from wandb.wandb_run import Run
except ModuleNotFoundError:
    # needed for test mocks, these tests shall be updated
    wandb, Run, RunDisabled = None, None, None  # type: ignore


class HKWandBLogger(WandbLogger):
    ### This is an ugly patch to fix the bug in wandb logger
    ### where `name` was set to project name instead of experiment name.
  def __init__(
        self,
        name: Optional[str] = None,
        save_dir: Optional[str] = None,
        offline: bool = False,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        version: Optional[str] = None,
        project: Optional[str] = None,
        log_model: Union[str, bool] = False,
        experiment: Union[Run, RunDisabled, None] = None,
        prefix: str = "",
        agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None,
        agg_default_func: Optional[Callable[[Sequence[float]], float]] = None,
        **kwargs: Any,
    ) -> None:
        if wandb is None:
            raise ModuleNotFoundError(
                "You want to use `wandb` logger which is not installed yet,"
                " install it with `pip install wandb`."  # pragma: no-cover
            )

        if offline and log_model:
            raise MisconfigurationException(
                f"Providing log_model={log_model} and offline={offline} is an invalid configuration"
                " since model checkpoints cannot be uploaded in offline mode.\n"
                "Hint: Set `offline=False` to log your model."
            )

        if log_model and not _WANDB_GREATER_EQUAL_0_10_22:
            rank_zero_warn(
                f"Providing log_model={log_model} requires wandb version >= 0.10.22"
                " for logging associated model metadata.\n"
                "Hint: Upgrade with `pip install --upgrade wandb`."
            )

        Logger.__init__(self, agg_key_funcs=agg_key_funcs, agg_default_func=agg_default_func)
        self._offline = offline
        self._log_model = log_model
        self._prefix = prefix
        self._experiment = experiment
        self._logged_model_time: Dict[str, float] = {}
        self._checkpoint_callback: Optional["ReferenceType[Checkpoint]"] = None
        # set wandb init arguments
        self._wandb_init: Dict[str, Any] = dict(
            name=name, ## This is where the patch is made
            project=project,
            id=version or id,
            dir=save_dir,
            resume="allow",
            anonymous=("allow" if anonymous else None),
        )
        self._wandb_init.update(**kwargs)
        # extract parameters
        self._save_dir = self._wandb_init.get("dir")
        self._name = self._wandb_init.get("name")
        self._id = self._wandb_init.get("id")
        # start wandb run (to create an attach_id for distributed modes)
        if _WANDB_GREATER_EQUAL_0_12_10:
            wandb.require("service")
            _ = self.experiment