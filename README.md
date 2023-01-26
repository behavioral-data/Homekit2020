Homekit2020
==============================

## Getting Started
------------

### Installation
1. Clone this repo: `git clone https://github.com/behavioral-data/Homekit2020.git`
2. cd into it:  `cd Homekit2020`
3. Build the conda environment: `make create_environment` (requires conda)
4. Install the src package: `conda activate Homekit2020; pip install -e .`

### Getting Our Data 
Navigate to https://www.synapse.org/#!Synapse:syn22803188/wiki/609492 to begin the approval process for access to the Homekit2020 dataset. Note that once you become a registered Synapse user it may take several business days for the Homekit2020 team to process your request. 

Once you have approval, follow these steps:
1. Install the Synapse CLI by following [these instructions](https://help.synapse.org/docs/Installing-Synapse-API-Clients.1985249668.html#InstallingSynapseAPIClients-CommandLine).
2. Download the zipped data with `synapse get syn32804645`
3. Create the data directory `mkdir data; unzip homekit2020neurips.zip -d data/processed`

### Running your first job 
This project was designed to be run primarily from the command line (although it _could_ be run from a notebook, e.g. by importing `src` ). You can run a simple job with:
``` bash
python src/models/train.py fit `# Main entry point` \
        --config configs/tasks/HomekitPredictFluPos.yaml `# Configures the task`\
        --config configs/models/CNNToTransformerClassifier.yaml `# Configures the model`\
        --data.train_path  $PWD/data/processed/split/audere_split_2020_02_10/train_7_day  `# Train data location`\
        --data.val_path $PWD/data/processed/split/audere_split_2020_02_10/eval_7_day  `# Validation data location`\
```

### Loading a Pretrained Model
TODO: Models need to be stored somewhere external
Pretrained models are located in the `models` subdirectory. To load a model for finetuning, pass the path to the model checkpoint to the training script like so:
``` bash
python src/models/train.py fit  \
        --trainer.resume_from_checkpoint PATH_TO_PRETRAINED
        --config configs/tasks/HomekitPredictFluPos.yaml \
        --config configs/models/CNNToTransformerClassifier.yaml \
        --data.train_path  $PWD/data/processed/split/audere_split_2020_02_10/train_7_day \
        --data.val_path  $PWD/data/processed/split/audere_split_2020_02_10/eval_7_day \
```

### Adding a new model
All models in this project should be based on [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).  This is done by subclassing `src.models.models.SensingModel` (or one of its derivatives, like `src.models.models.ClassificationModel`). All that's necessary is overriding the `forward` method, and optionally `__init__`:

```python

from pytorch_lightning.utilities.cli import MODEL_REGISTRY

class ResNet(ClassificationModel):
    def __init__(self,
                 n_layers: int = 3,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        
        self.base_model = ResNet(n_layers)
        self.criterion = nn.CrossEntropyLoss() 
        self.save_hyperparameters()
    
    def forward(self, x,labels):
        preds = self.base_model(x)
        loss =  self.criterion(preds,labels)
        return loss, preds
```
The model can now be called from the command line (or a config):

``` bash
python src/models/train.py fit `# Main entry point` \
        --config src/data/task_configs/PredictFluPos.yaml `# Configures the task`\
        --model ResNet\
        --model.n_layers 10\
        --data.train_path $PWD/data/debug/petastorm_datasets/debug `# Train data location`\
        --data.val_path $PWD/data/debug/petastorm_datasets/debug `# Validation data location`\
```


### Extending a Pretrained Model 
Since all models are built with pytorch, it's easy to extend pretrained models.  Let's say that you had a model that had been pretrained for regression, but which you wanted to use for classification. All that's needed is to structure your new model similarly to the old, and let the `state_dict` do the rest:

```python
 
class MyRegressionModel(RegressionModel):
    def __init__(self,
                 n_layers: int = 3,
                 n_outputs = 10,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        
        self.encoder = SomeBigEncoder(n_layers)
        self.head = RegressionModule(self.encoder.d_model, self.encoder.final_length, n_outputs)
        self.criterion = nn.MSELoss() 
        self.save_hyperparameters()
    
    def forward(self, x,labels):
        x = self.encoder(x)
        preds = self.head(x)
        loss = self.criterion(preds,loss)

        return loss, preds

 
class MyClassificationModel(ClassificationModel):
    def __init__(self,
                 n_layers: int = 3
                 **kwargs) -> None:

        super().__init__(**kwargs)
        
        # All we need to do is create a new head. Since the dimensions are different,
        # the parameters from the pretrained model will be ignored
        self.head = ClassificationModule(self.encoder.d_model, 
                                         self.encoder.final_length, 2)

```

### Adding a new task
Tasks are repsonsible for setting up Dataloaders and calculating evaluation metrics. All tasks are defined in  `src/models/tasks.py`, and should subclass `ActivityTask` (which in turn ultimately subclasses `pl.LightningDataModule`). 

#### Lablers
All tasks must have a `Labeler`, which is a callable object with the following method signature:
```python
def __call__(self,participant_id,start_date,end_date):
        ... # Return the label for this window

```
Some lablers (like `PredictSurveyClause`) take arguments that modify the behavior. Examples are available in `src/models/lablers.py`. 

Let's look at an example `Task` and `Labler`:
```python

class ClauseLabler(object):
    def __init__(self, survey_respones, clause):
        self.clause = clause
        self.survey_responses = survey_respones
        self.survey_responses["_date"] = self.survey_responses["timestamp"].dt.normalize()
        self.survey_responses["_dummy"] = True
        self.survey_lookup = self.survey_responses\
                                 .reset_index()\
                                 .drop_duplicates(subset=["participant_id","_date"],keep="last")\
                                 .set_index(["participant_id","_date"])\
                                 .query(self.clause)\
                                 ["_dummy"]\
                                 .to_dict()

    def __call__(self,participant_id,start_date,end_date):
        result = self.survey_lookup.get((participant_id,end_date.normalize()),False)
        return int(result)


class PredictSurveyClause(ActivityTask,ClassificationMixin):
    """Predict the whether a clause in the onehot
       encoded surveys is true for a given day. 
       
       For a sense of what kind of logical clauses are
       supported, check out:
    
       https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html"""

    def __init__(self, clause: str, 
                       activity_level: str = "minute", 
                       fields: List[str] = DEFAULT_FIELDS, 
                       survey_path: Optional[str] = None,
                       **kwargs):
        self.clause = clause
        self.survey_responses = load_processed_table("daily_surveys_onehot",path=survey_path).set_index("participant_id")
        self.labler = ClauseLabler(self.survey_responses,self.clause)
        ActivityTask.__init__(self, fields=fields, activity_level=activity_level,**kwargs)
        ClassificationMixin.__init__(self)
    
    def get_labler(self):
        return self.labler

    def get_description(self):
        return self.__doc__

```

With this setup, we can train a model that predicts if an arbitrary boolean combination of survey responses is true:

```bash
python src/models/train.py fit \
        --data PredictSurveyClause `# Tell the script which task you want to use`\
        --data.clause 'symptom_severity__cough_q_3 > 0' `# Predict severe cough`\
        --data.survey_path PATH_TO_ONEHOT_SURVEY_CSV\
        --data.train_path $PWD/data/debug/petastorm_datasets/debug `# Train data location`\
        --data.val_path $PWD/data/debug/petastorm_datasets/debug `# Validation data location`\
        --model ResNet
```
###  [Optional] Weights and Biases Integration:

By default this project integrates with Weights and Biases. If you would like to ignore this integration and use some other logging infrasturcture, run commands with the `--no_wandb` flag.

In order to set up this integration, add the following to `.env` (and, of course, install wandb):
```
WANDB_USERNAME=<your username>
WANDB_PROJECT=<the name of the WandB project you want to save results to>
```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
