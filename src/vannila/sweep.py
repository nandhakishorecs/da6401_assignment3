# import tensorflow as tf
# import wandb                                # type: ignore
# import yaml                                 # type: ignore

# import warnings
# warnings.filterwarnings('ignore')

# import sys 
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data')))

# from data_loader import * 
# from network import * 

# # Set random seed for reproducibility
# tf.random.set_seed(42)


# SWEEP_NAME = 'da6401_assignment2'
# EXPERIMENT_COUNT = 120

# with open("sweep_config.yml", "r") as file:
#     sweep_config = yaml.safe_load(file)

# sweep_id = wandb.sweep(sweep_config,project=SWEEP_NAME)

# def do_sweep(): 
#     wandb.init(project = 'da6401_assignment3_vannila_sweep')
#     config = wandb.config 

#     wandb.run.name = f''
    
    
#     try:
#         model.train(
    
#         )
#     except Exception as e:
#         print(f"Training failed with error: {str(e)}")
#         raise

#     wandb.finish()

# if __name__ == '__main__': 
#     wandb.agent(sweep_id, function = do_sweep, count = EXPERIMENT_COUNT)