import wandb

# Project that the run is recorded to
project = "test-wandb-project"

# Dictionary with hyperparameters
config = {
    'epochs' : 10,
    'lr' : 0.01
}

with wandb.init(project=project,
                entity=None,
                name="foo_run",
                tags=[],
                config=config) as run:
    # Training code here
    # Log values to W&B with run.log()
    for step in range(0, 10):
        acc = step*0.1
        loss = 0.1 - step * 0.01
        run.log({"accuracy": acc, "loss": loss})
