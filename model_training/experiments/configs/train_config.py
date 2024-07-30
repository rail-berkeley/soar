from ml_collections import ConfigDict


def get_config(config_string):
    base_real_config = dict(
        batch_size=512,
        num_steps=int(1001000),
        log_interval=1000,
        eval_interval=25000,
        save_interval=25000,
        num_val_trajs=8,
        num_val_batches=8,
        save_dir="~/jaxrl_log",
        resume_path="",
        seed=42,
    )

    base_data_config = dict(
        # action_merge_horizon=2,
        shuffle_buffer_size=25000,
        augment=True,
        augment_next_obs_goal_differently=False,
        augment_kwargs=dict(
            random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
            random_brightness=[0.2],
            random_contrast=[0.8, 1.2],
            random_saturation=[0.8, 1.2],
            random_hue=[0.1],
            augment_order=[
                "random_resized_crop",
                "random_brightness",
                "random_contrast",
                "random_saturation",
                "random_hue",
            ],
        ),
    )

    possible_structures = {
        "gc_bc_offline_bridge": ConfigDict(
            dict(
                agent="gc_bc",
                agent_kwargs=dict(
                    network_kwargs=dict(
                        hidden_dims=(
                            256,
                            256,
                            256,
                            256,
                        ),  # diff: bridge release use 3 * 256
                        dropout_rate=0,  # diff: bridge release use 0.1
                    ),
                    policy_kwargs=dict(
                        tanh_squash_distribution=True,  # diff: bridge release use MSE head
                        std_parameterization="fixed",
                        fixed_std=[1, 1, 1, 1, 1, 1, 1],
                    ),
                    early_goal_concat=True,
                    shared_goal_encoder=True,
                    use_proprio=False,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    decay_steps=int(2e6),
                ),
                dataset_kwargs=dict(
                    goal_relabeling_strategy="geometric",
                    goal_relabeling_kwargs=dict(reached_proportion=0.0, discount=0.98),
                    normalization_type="normal",
                    **base_data_config,
                ),
                encoder="resnetv1-34-bridge",  # diff: bridge release use resnet-50
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                **base_real_config,
            )
        ),
    }

    return possible_structures[config_string]
