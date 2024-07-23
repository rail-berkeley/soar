"""
Contains goal relabeling and reward logic written in TensorFlow.

Each relabeling function takes a trajectory with keys "observations",
"next_observations", and "terminals". It returns a new trajectory with the added
keys "goals", "rewards", and "masks". Keep in mind that "observations" and
"next_observations" may themselves be dictionaries, and "goals" must match their
structure.

"masks" determines when the next Q-value is masked out. Typically this is NOT(terminals).
Note that terminal may be changed when doing goal relabeling.
"""

import tensorflow as tf


def uniform(traj, *, reached_proportion, discount):
    """
    Relabels with a true uniform distribution over future states. With
    probability reached_proportion, observations[i] gets a goal
    equal to next_observations[i].  In this case, the reward is 0. Otherwise,
    observations[i] gets a goal sampled uniformly from the set
    next_observations[i + 1:], and the reward is -1.
    """
    traj_len = tf.shape(traj["terminals"])[0]

    # select a random future index for each transition i in the range [i + 1, traj_len)
    rand = tf.random.uniform([traj_len])
    low = tf.cast(tf.range(traj_len) + 1, tf.float32)
    high = tf.cast(traj_len, tf.float32)
    goal_idxs = tf.cast(rand * (high - low) + low, tf.int32)

    # sometimes there are floating-point errors that cause an out-of-bounds
    goal_idxs = tf.minimum(goal_idxs, traj_len - 1)

    # select a random proportion of transitions to relabel with the next observation
    goal_reached_mask = tf.random.uniform([traj_len]) < reached_proportion

    # the last transition must be goal-reaching
    goal_reached_mask = tf.logical_or(
        goal_reached_mask, tf.range(traj_len) == traj_len - 1
    )

    # make goal-reaching transitions have an offset of 0
    goal_idxs = tf.where(goal_reached_mask, tf.range(traj_len), goal_idxs)

    # select goals
    traj["goals"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        traj["next_observations"],
    )

    # reward is 0 for goal-reaching transitions, -1 otherwise
    traj["rewards"] = tf.cast(tf.where(goal_reached_mask, 0, -1), tf.int32)

    # add masks
    traj["masks"] = tf.logical_not(goal_reached_mask)

    # add distances to goal
    traj["goal_dists"] = goal_idxs - tf.range(traj_len)

    # We'll also need to add mc_returns
    # Since we have a prior on the reward values, computing discounted sums is easier
    traj["mc_returns"] = (
        -1
        * (tf.math.pow(discount, tf.cast(traj["goal_dists"], tf.float32)) - 1)
        / (discount - 1)
    )

    return traj


def last_state_upweighted(traj, *, reached_proportion):
    """
    A weird relabeling scheme where the last state gets upweighted. For each
    transition i, a uniform random number is generated in the range [i + 1, i +
    traj_len). It then gets clipped to be less than traj_len. Therefore, the
    first transition (i = 0) gets a goal sampled uniformly from the future, but
    for i > 0 the last state gets more and more upweighted.
    """
    traj_len = tf.shape(traj["terminals"])[0]

    # select a random future index for each transition
    offsets = tf.random.uniform(
        [traj_len],
        minval=1,
        maxval=traj_len,
        dtype=tf.int32,
    )

    # select random transitions to relabel as goal-reaching
    goal_reached_mask = tf.random.uniform([traj_len]) < reached_proportion
    # last transition is always goal-reaching
    goal_reached_mask = tf.logical_or(
        goal_reached_mask, tf.range(traj_len) == traj_len - 1
    )

    # the goal will come from the current transition if the goal was reached
    offsets = tf.where(goal_reached_mask, 0, offsets)

    # convert from relative to absolute indices
    indices = tf.range(traj_len) + offsets

    # clamp out of bounds indices to the last transition
    indices = tf.minimum(indices, traj_len - 1)

    # select goals
    traj["goals"] = tf.nest.map_structure(
        lambda x: tf.gather(x, indices),
        traj["next_observations"],
    )

    # reward is 0 for goal-reaching transitions, -1 otherwise
    traj["rewards"] = tf.cast(tf.where(goal_reached_mask, 0, -1), tf.int32)

    # add masks
    traj["masks"] = tf.logical_not(goal_reached_mask)

    # add distances to goal
    traj["goal_dists"] = goal_idxs - tf.range(traj_len)

    return traj


def geometric(traj, *, reached_proportion, commanded_goal_proportion=-1.0, discount):
    """
    Relabels with a geometric distribution over future states. With
    probability reached_proportion, observations[i] gets a goal
    equal to next_observations[i].  In this case, the reward is 0. Otherwise,
    observations[i] gets a goal sampled geometrically from the set
    next_observations[i + 1:], and the reward is -1.
    """
    traj_len = tf.shape(traj["terminals"])[0]

    # geometrically select a future index for each transition i in the range [i + 1, traj_len)
    arange = tf.range(traj_len)
    is_future_mask = tf.cast(arange[:, None] < arange[None], tf.float32)
    d = discount ** tf.cast(arange[None] - arange[:, None], tf.float32)

    probs = is_future_mask * d
    # The indexing changes the shape from [seq_len, 1] to [seq_len]
    goal_idxs = tf.random.categorical(
        logits=tf.math.log(probs), num_samples=1, dtype=tf.int32
    )[:, 0]

    # select a random proportion of transitions to relabel with the next observation
    goal_reached_mask = tf.random.uniform([traj_len]) < reached_proportion

    # the last transition must be goal-reaching
    goal_reached_mask = tf.logical_or(
        goal_reached_mask, tf.range(traj_len) == traj_len - 1
    )

    # make goal-reaching transitions have an offset of 0
    goal_idxs = tf.where(goal_reached_mask, tf.range(traj_len), goal_idxs)

    # select goals
    traj["goals"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        traj["next_observations"],
    )

    # reward is 0 for goal-reaching transitions, -1 otherwise
    traj["rewards"] = tf.cast(tf.where(goal_reached_mask, 0, -1), tf.int32)

    # Now we will perform the logic for sampling commanded goals
    if commanded_goal_proportion != -1.0:  # if commanded goals are in the dataset
        commanded_goal_mask = tf.random.uniform([traj_len]) < commanded_goal_proportion
        commanded_goal_mask = tf.logical_and(
            commanded_goal_mask, tf.range(traj_len) != traj_len - 1
        )  # last transition must remain goal reaching
        traj["goals"]["image"] = tf.where(
            commanded_goal_mask[..., None, None, None],
            traj["commanded_goals"],
            traj["goals"]["image"],
        )
        traj["rewards"] = tf.where(commanded_goal_mask, -1, traj["rewards"])

    # add masks
    # traj["masks"] = tf.logical_not(traj["terminals"]) # I don't think this line is correct
    if commanded_goal_proportion != -1.0:
        traj["masks"] = tf.logical_or(
            tf.logical_not(goal_reached_mask), commanded_goal_mask
        )
    else:
        traj["masks"] = tf.logical_not(goal_reached_mask)

    # add distances to goal
    traj["goal_dists"] = goal_idxs - tf.range(traj_len)

    # We'll also need to add mc_returns
    # Since we have a prior on the reward values, computing discounted sums is easier
    traj["mc_returns"] = 0 * tf.math.pow(
        discount, tf.cast(traj["goal_dists"], tf.float32)
    ) + -1 * (tf.math.pow(discount, tf.cast(traj["goal_dists"], tf.float32)) - 1) / (
        discount - 1
    )
    if commanded_goal_proportion != -1.0:
        traj["mc_returns"] = tf.where(
            commanded_goal_mask, -50.0, traj["mc_returns"]
        )  # hardcoding, but we don't have access to RL discount factor so..

    return traj


def delta_goals(traj, *, goal_delta, discount = 0.99):
    """
    Relabels with a uniform distribution over future states in the range [i +
    goal_delta[0], min{traj_len, i + goal_delta[1]}). Truncates trajectories to
    have length traj_len - goal_delta[0].
    """
    traj_len = tf.shape(traj["terminals"])[0]

    # add the last observation (which only exists in next_observations) to get
    # all the observations
    all_obs = tf.nest.map_structure(
        lambda obs, next_obs: tf.concat([obs, next_obs[-1:]], axis=0),
        traj["observations"],
        traj["next_observations"],
    )
    all_obs_len = traj_len + 1

    # current obs should only come from [0, traj_len - goal_delta[0])
    curr_idxs = tf.range(traj_len - goal_delta[0])

    # select a random future index for each transition i in the range [i + goal_delta[0], min{all_obs_len, i + goal_delta[1]})
    rand = tf.random.uniform([traj_len - goal_delta[0]])
    low = tf.cast(curr_idxs + goal_delta[0], tf.float32)
    high = tf.cast(tf.minimum(all_obs_len, curr_idxs + goal_delta[1]), tf.float32)
    goal_idxs = tf.cast(rand * (high - low) + low, tf.int32)

    # very rarely, floating point errors can cause goal_idxs to be out of bounds
    goal_idxs = tf.minimum(goal_idxs, all_obs_len - 1)

    # it doesn't make sense to sample the current state as the goal
    # this operation also slightly upweights sampling of the next state as the goal, 
    # which is good for RL
    goal_idxs = tf.maximum(goal_idxs, curr_idxs + 1)

    traj_truncated = tf.nest.map_structure(
        lambda x: tf.gather(x, curr_idxs),
        traj,
    )

    # select goals
    traj_truncated["goals"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        all_obs,
    )

    # add distances to goal
    traj_truncated["goal_dists"] = goal_idxs - curr_idxs

    # add rewards
    goal_is_next_obs_mask = traj_truncated["goal_dists"] == 1
    traj_truncated["rewards"] = tf.where(goal_is_next_obs_mask, 0.0, -1.0)

    # add masks
    traj_truncated["masks"] = tf.where(goal_is_next_obs_mask, 0.0, 1.0)

    # add mc_returns
    traj_truncated["mc_returns"] = -1 * (tf.math.pow(discount, tf.cast(traj_truncated["goal_dists"] - 1, tf.float32)) - 1) / (discount - 1)

    return traj_truncated


GOAL_RELABELING_FUNCTIONS = {
    "uniform": uniform,
    "last_state_upweighted": last_state_upweighted,
    "geometric": geometric,
    "delta_goals": delta_goals,
}
