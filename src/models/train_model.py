#pylint: disable=import-error
from tensorflow.keras.callbacks import EarlyStopping

from src.models.neural_baselines import create_neural_model
from src.utils import get_logger
logger = get_logger()

def train_neural_baseline(model_name,task,
                         n_epochs=10,
                         early_stopping=True,
                         pos_class_weight = 100,
                         neg_class_weight = 1,
                         val_split = 0.15,
                         use_wandb=True):

                
    
    # Annoyingly need to load all of this into RAM:
    train_X, train_y = task.get_train_dataset().to_stacked_numpy()
    eval_X, eval_y  = task.get_eval_dataset().to_stacked_numpy()

    infer_example = train_X[0]
    n_timesteps, n_features = infer_example.shape
    
    model = create_neural_model(model_name, n_timesteps,n_features)
    
    callbacks = []
    if early_stopping:
        early_stopping_monitor = EarlyStopping(
                    monitor='val_loss',
                    min_delta=0,
                    patience=10,
                    verbose=0,
                    mode='min',
                    baseline=None,
                    restore_best_weights=True
                )
        callbacks.append(early_stopping_monitor)
    if use_wandb:
        from wandb.keras import WandbCallback
        import wandb
        wandb.init(project="flu",
                   entity="mikeamerrill",
                   config={"n_epochs": n_epochs,
                           "pos_class_weight": pos_class_weight,
                           "neg_class_weight": neg_class_weight,
                           "model_type":model_name})
                           
        callbacks.append(WandbCallback())

    logger.info(f"Training {model_name}")
    model.fit(train_X, train_y, 
            class_weight = {1: pos_class_weight, 0: neg_class_weight}, 
            epochs=n_epochs, validation_split=val_split, 
            callbacks = callbacks, verbose=1)
    
    logger.info(f"Training complete. Running evaluation...")
    pred_prob = model.predict(eval_X, verbose=0)
    results = task.evaluate_results(pred_prob,eval_y)
    logger.info("Eval results...")
    logger.info(results)

    