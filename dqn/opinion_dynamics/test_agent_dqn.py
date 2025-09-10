# test_agent_dqn.py
import pytest
import numpy as np
import torch
import random  # <-- fix the "NameError: name 'random' is not defined"
import gym
from unittest.mock import MagicMock

# Suppose your agent code is in agent_dqn.py
from dqn.opinion_dynamics.opinion_dqn import AgentDQN
from dqn.opinion_dynamics.utils.env_setup import build_environment


@pytest.fixture
def actual_env():
    return build_environment()

@pytest.fixture
def actual_validation_env():
    # For simplicity, we use the same environment as validation.
    return build_environment()

@pytest.fixture
def agent_config():
    """A minimal configuration for testing the agent using the real environment.
    Note: The actual environment is the NetworkGraph with 4 agents.
    """
    config = {
        "experiment": "opinion_agent_dqn",
        "epochs_to_train": 2,
        "environment": ["opinion_net"],
        "agent_params": {
            "agent": "AgentDQN",
            "args": {
                "train_step_cnt": 10,
                "validation_enabled": True,
                "validation_step_cnt": 6,
                "validation_epsilon": 0.001,
                "replay_start_size": 2,
                "batch_size": 2,
                "training_freq": 1,
                "target_model_update_freq": 2,
                "loss_fcn": "mse_loss",
                "gamma": 0.90,
                "epsilon": {
                    "start": 1.0,
                    "end": 0.01,
                    "decay": 25000
                }
            }
        },
        "estimator": {
            "model": "OpinionNet",
            "args": {
                "lin_hidden_size": 16
            }
        },
        "optim": {
            "name": "Adam",
            "args": {
                "lr": 0.00025,
                "eps": 0.0003125
            }
        },
        "replay_buffer": {
            "max_size": 50,
            "action_dim": 1,
            "n_step": 0
        }
    }
    return config

###############################################################################
# 2. Tests
###############################################################################
def test_agent_initialization(actual_env, actual_validation_env, agent_config):
    """Test that the agent initializes correctly and reads the config properly."""
    agent = AgentDQN(
        train_env=actual_env,
        validation_env=actual_validation_env,
        config=agent_config
    )
    # For your actual environment, observation space shape is (num_agents,), which is (4,)
    assert agent.in_features == 4, "Observation dimension should be 4"
    assert agent.gamma == 0.90
    assert agent.replay_buffer.max_size == 50
    assert agent.betas == [0, 1]

def test_select_action(actual_env, agent_config):
    """Check that select_action always returns a 3-tuple and that the action shape is as expected."""
    agent = AgentDQN(train_env=actual_env, validation_env=actual_env, config=agent_config)
    # Get an initial state from the actual environment.
    state, _ = actual_env.reset()
    state_t = torch.tensor(state, dtype=torch.float32)
    
    # Exploitation path.
    action_np, betas, max_q = agent.select_action(state_t, epsilon=0.0, random_action=False)
    # The action should have shape (4,) (one per agent).
    assert action_np.shape in [(4,), (4,1)], f"Unexpected action shape: {action_np.shape}"
    assert isinstance(max_q, float)
    
    # Exploration path.
    action_rand, betas_rand, max_q_rand = agent.select_action(state_t, epsilon=1.0, random_action=False)
    assert action_rand.shape in [(4,), (4,1)]

def test_run_single_episode(actual_env, agent_config):
    """
    Run a single training epoch using the actual environment.
    Check that the replay buffer is populated and that we get some epoch stats.
    """
    agent_config["agent_params"]["args"]["train_step_cnt"] = 5
    agent = AgentDQN(train_env=actual_env, validation_env=actual_env, config=agent_config)
    
    # Force epsilon=1 so we always explore.
    agent.epsilon_by_frame = lambda x: 1.0
    
    ep_train_stats = agent.train_epoch()
    # Since we use the real environment, we might not see complete episodes,
    # but we at least expect some transitions stored.
    assert len(agent.replay_buffer) > 0, "Replay buffer should have stored transitions"

def test_model_learn(actual_env, agent_config):
    """
    Populate the replay buffer with minimal transitions using the actual environment,
    then run a single model_learn step and ensure the model parameters update.
    """
    agent = AgentDQN(train_env=actual_env, validation_env=actual_env, config=agent_config)
    
    # Populate the replay buffer.
    for _ in range(agent.batch_size):
        s, _ = actual_env.reset()
        s = np.array(s, dtype=np.float32)
        # We'll store the chosen beta as 0 (as an integer index).
        chosen_beta = np.array([0], dtype=np.int64)
        r = 0.0
        s_prime, _ = actual_env.reset()
        s_prime = np.array(s_prime, dtype=np.float32)
        done = False
        agent.replay_buffer.append(s, chosen_beta, r, s_prime, done)
    
    old_params = [p.clone().detach() for p in agent.policy_model.parameters()]
    batch = agent.replay_buffer.sample(agent.batch_size)
    loss_val = agent.model_learn(batch, debug=True)
    assert isinstance(loss_val, float), "model_learn should return a float loss"
    
    # Check that at least one parameter has changed.
    for old_p, new_p in zip(old_params, agent.policy_model.parameters()):
        if not torch.equal(old_p, new_p):
            break
    else:
        pytest.fail("Model parameters did not update after learning")

def test_train_validation_loop(actual_env, actual_validation_env, agent_config):
    """
    Integration test: run train() for a couple epochs using the actual environment;
    confirm that training_stats and validation_stats are recorded.
    """
    agent_config["agent_params"]["args"]["train_step_cnt"] = 8
    agent_config["agent_params"]["args"]["validation_step_cnt"] = 4
    agent_config["epochs_to_train"] = 2
    
    agent = AgentDQN(train_env=actual_env, validation_env=actual_validation_env, config=agent_config)
    agent.train(train_epochs=2)
    assert len(agent.training_stats) == 2, "Expected 2 training stats entries"
    assert len(agent.validation_stats) == 2, "Expected 2 validation stats entries"
    
    for st in agent.training_stats:
        assert "episode_rewards" in st
        assert "episode_discounted_rewards" in st
        assert "policy_trained_times" in st