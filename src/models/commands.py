import json
from random import choice
import click
from src.utils import read_yaml

def validate_dataset_args(ctx, param, value):
    try:
        return read_yaml(value)
    except FileNotFoundError:
        try:
            return json.loads(value)
        except json.decoder.JSONDecodeError:
            raise click.BadParameter('dataset_args needs to be either a json string or a path to a config .yaml')

huggingface_options = [
     click.Option(('--n_epochs',), default = 10, help='Number of training epochs'),
     click.Option(('--hidden_size',), default = 768),
     click.Option(('--num_attention_heads',), default = 4),
     click.Option(('--num_hidden_layers',), default = 4),
     click.Option(('--max_length',), default = 24*60+1),
     click.Option(('--max_position_embeddings',), default = 2048),
     # Training Args:
     click.Option(('--pos_class_weight',), default = 1),
     click.Option(('--neg_class_weight',), default = 4),
     click.Option(('--train_batch_size',), default = 20),
     click.Option(('--eval_batch_size',), default = 60),
     click.Option(('--no_early_stopping',), is_flag=True),
     click.Option(('--warmup_steps',), default = 500),
     click.Option(('--weight_decay',), default = 0.1),
     click.Option(('--eval_frac',), default = None, type=float),
     click.Option(('--learning_rate',), default = 5e-5),
     click.Option(('--sinu_position_encoding',), is_flag=True, default=False),
     click.Option(('--classification_threshold',), default = 0.5),
]

universal_options = [
    click.Option(('--no_wandb',), is_flag=True),
    click.Option(('--notes',), type=str, default=None),
    click.Option(('--dataset_args',), default=None,callback=validate_dataset_args),
    click.Option(('--activity_level',),type=click.Choice(["day","minute"]), default="minute")
]

class BaseCommand(click.Command):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = self.params + universal_options

class HuggingFaceCommand(BaseCommand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = self.params + huggingface_options 

