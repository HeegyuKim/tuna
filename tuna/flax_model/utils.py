import jax

checkpoint_policies = dict(
    everything_saveable=jax.checkpoint_policies.everything_saveable,
    nothing_saveable=jax.checkpoint_policies.nothing_saveable,
    dots_saveable=jax.checkpoint_policies.dots_saveable,
    checkpoint_dots=jax.checkpoint_policies.checkpoint_dots,
    dots_with_no_batch_dims_saveable=jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
    checkpoint_dots_with_no_batch_dims=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
    save_anything_except_these_names=jax.checkpoint_policies.save_anything_except_these_names,
    save_any_names_but_these=jax.checkpoint_policies.save_any_names_but_these,
    save_only_these_names=jax.checkpoint_policies.save_only_these_names,
    save_from_both_policies=jax.checkpoint_policies.save_from_both_policies
)

def get_gradient_checkpoint_policy(name):
    """
    The get_gradient_checkpoint_policy function is a helper function that returns the gradient checkpoint policy
        specified by the name parameter.

    :param name: Select the checkpoint policy from the dictionary
    :return: A function that is used in the jax

    """
    return checkpoint_policies[name]