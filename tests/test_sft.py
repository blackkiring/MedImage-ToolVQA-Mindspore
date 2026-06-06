from medimage_toolvqa_ms.sft import MindSporeSFTTrainer


def test_sft_trainer_dry_run_uses_dataset_config():
    trainer = MindSporeSFTTrainer(
        {
            "training": {
                "model_name_or_path": "model",
                "train_records": "train.jsonl",
                "eval_records": "eval.jsonl",
                "output_dir": "outputs/sft",
            },
            "sft_data": {"image_root": "images"},
            "mindspore": {"device_target": "Ascend"},
        }
    )

    summary = trainer.dry_run_summary()

    assert summary["train_records"] == "train.jsonl"
    assert summary["eval_records"] == "eval.jsonl"
    assert summary["image_root"] == "images"
    assert summary["status"] == "mindspore_sft_training_pending"
