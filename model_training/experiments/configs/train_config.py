import os
from ml_collections import ConfigDict

from jaxrl_m.agents.continuous.cql import (
    get_default_config as get_continuous_cql_config,
)
from jaxrl_m.agents.discrete.cql import get_default_config as get_discrete_cql_config


def get_config(config_string):
    base_real_config = dict(
        batch_size=512,
        num_steps=int(1001000),
        log_interval=1000,
        eval_interval=25000,
        save_interval=25000,
        num_val_trajs=8,
        num_val_batches=8,
        save_dir="gs://paul-v4-central2-b/jaxrl_log",
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
        "calql_offline_bridge": ConfigDict(
            dict(
                agent="calql",
                agent_kwargs=get_continuous_cql_config(
                    updates=dict(
                        policy_kwargs=dict(
                            tanh_squash_distribution=True,
                            std_parameterization="exp",
                        ),
                        distributional_critic=False,
                        distributional_critic_kwargs={
                            "q_min": -50.0,
                            "q_max": 0.0,
                            "num_bins": 64,
                        },
                        critic_network_type="ptr_critic",
                        critic_network_kwargs={
                            "hidden_dims": [256, 256, 256, 256],
                            "activate_final": True,
                            "use_layer_norm": True,
                            "use_group_norm": False,
                        },
                        policy_network_kwargs={
                            "hidden_dims": [256, 256, 256, 256],
                            "activate_final": True,
                            "use_layer_norm": True,
                            "use_group_norm": False,
                        },
                        actor_optimizer_kwargs={
                            "learning_rate": 3e-4,
                            "warmup_steps": 2000,
                            "clip_grad_norm": 1,
                        },
                        critic_optimizer_kwargs={
                            "learning_rate": 3e-4,
                            "warmup_steps": 2000,
                            "clip_grad_norm": 1,
                        },
                        temperature_optimizer_kwargs={
                            "learning_rate": 3e-4,
                            "clip_grad_norm": 1,
                        },
                        early_goal_concat=True,
                        goal_conditioned=True,
                        gc_kwargs={
                            "negative_proportion": 0.3,
                        },
                        shared_goal_encoder=True,
                        actor_use_proprio=False,
                        critic_use_proprio=False,
                        cql_max_target_backup=True,
                        shared_encoder=False,
                        discount=0.98,
                        cql_alpha=1.0,
                        cql_n_actions=4,
                        critic_ensemble_size=10,
                        critic_subsample_size=2,
                        bc_loss_weight=0.1,
                        dr3_coefficient=1.0,
                        stop_actor_encoder_gradient=False,
                        stop_critic_encoder_gradient=False,
                    )
                ).to_dict(),
                dataset_kwargs=dict(
                    goal_relabeling_strategy="geometric",
                    goal_relabeling_kwargs=dict(reached_proportion=0.2, discount=0.98),
                    normalization_type="tanh",
                    dataset_contains_commanded_goals=False,
                    action_clip_delta=0.001,
                    **base_data_config,
                ),
                pretrained_loaders=[],
                encoder="resnetv1-34-bridge",  # "resnetv1-18"
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                # pretrained_loaders=[
                #     (
                #         "calql_dr3_resnet_gc",
                #         dict(
                #             checkpoint_path="/nfs/nfs2/users/paulzhou/checkpoints/calql-dr-3/checkpoint_700000",
                #         ),
                #     )
                # ],
                # pretrained_loaders=[
                #     (
                #         "bigvision_resnet_gc",
                #         dict(
                #             modules=["modules_actor"],
                #             checkpoint_path="/nfs/nfs2/users/paulzhou/checkpoints/big_transfer/BiT-M-R50x1.npz",
                #             clone_new_weights=True,
                #         ),
                #     )
                # ],
                # encoder="resnetv2-50-1",
                # encoder_kwargs=dict(),
                **base_real_config,
            )
        ),
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
                    normalization_type="tanh_normal",
                    dataset_contains_commanded_goals=False,
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
