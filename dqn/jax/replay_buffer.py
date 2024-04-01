from functools import partial
from abc import ABC, abstractmethod

from jax import jit, random, tree_map, vmap


class BaseReplayBuffer(ABC):
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
    ) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    @partial(jit, static_argnums=(0))
    def add(
        self,
        buffer_state: dict,
        experience: tuple,
        idx: int,
    ):
        state, action, reward, next_state, done = experience
        idx = idx % self.buffer_size

        buffer_state["states"] = buffer_state["states"].at[idx].set(state)
        buffer_state["actions"] = buffer_state["actions"].at[idx].set(action)
        buffer_state["rewards"] = buffer_state["rewards"].at[idx].set(reward)
        buffer_state["next_states"] = (
            buffer_state["next_states"].at[idx].set(next_state)
        )
        buffer_state["dones"] = buffer_state["dones"].at[idx].set(done)

        return buffer_state

    @abstractmethod
    def sample(self):
        pass


class UniformReplayBuffer(BaseReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
    ) -> None:
        super(UniformReplayBuffer, self).__init__(buffer_size, batch_size)

    @partial(jit, static_argnums=(0))
    def sample(
        self,
        key: random.PRNGKey,
        buffer_state: dict,
        current_buffer_size: int,
    ):
        """
        Samples a random experience from the replay buffer using
        the uniform distribution.

        Args:
            key (random.PRNGKey): the random key used to sample the buffer
            buffer (dict): the buffer to sample experiences from,
                keys: "states", "actions", "rewards", "next_states", "dones"
            current_buffer_size (int): the number of experiences currently stored in the buffer

        returns:
            dict[str: jnp.ndarray]: A dictionary with keys "states", "actions", "next_states",
            "dones", "rewards"
        """

        @partial(vmap, in_axes=(0, None))
        def sample_batch(indexes, buffer):
            return tree_map(lambda x: x[indexes], buffer)

        key, subkey = random.split(key)
        indexes = random.randint(
            subkey,
            shape=(self.batch_size,),
            minval=0,
            maxval=current_buffer_size,
        )
        experiences = sample_batch(indexes, buffer_state)

        return experiences, subkey
