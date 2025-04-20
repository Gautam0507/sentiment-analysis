from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig


def start_training():
    print("Starting training...")
    tesnorboard_config = TensorBoardOutputConfig(
        s3_output_path="s3://your-bucket-name/tensorboard-output",
        container_local_output_path="/opt/ml/output/tensorboard",
    )
    estimator = PyTorch(
        entry_point="train.py",
        source_dir="training",
        role="my-new-role",
        py_version="py311",
        instance_count=1,
        instance_type="ml.g5.xlarge",
        hyperparameters={
            "batch_size": 32,
            "epochs": 25,
        },
        tensorboard_config=tensorboard_config,
    )

    estimator.fit(
        {
            "training": "s3://your-bucket-name/dataset/train",
            "validation": "s3://your-bucket-name/dataset/dev",
            "test": "s3://your-bucket-name/dataset/test",
        }
    )


if __name__ == "__main__":
    start_training()
