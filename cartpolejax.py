import jax.numpy as jnp
from jax import jit, lax
import jax.random as random

# --- Utility Functions ---

def remap_angle(theta):
    return jnp.mod(theta + jnp.pi, 2. * jnp.pi) - jnp.pi

def loss(state, sig=None):
    if sig is None:
        sig = jnp.full_like(state, 0.5)
    exponent = -jnp.sum((state**2) / (2.0 * sig**2))
    return 1 - jnp.exp(exponent)

# --- JAX-Compatible CartPole Environment ---

class JAXCartPole:
    def __init__(self):
        self.params = {
            "pole_length": 0.5,
            "pole_mass": 0.5,
            "cart_mass": 0.5,
            "mu_c": 0.001,
            "mu_p": 0.001,
            "gravity": 9.8,
            "max_force": 20.0,
            "delta_time": 0.1,
            "sim_steps": 1,
        }
        self._state = jnp.array([0.0, 0.0, jnp.pi, 0.0])  # [x, x_dot, theta, theta_dot]

    def reset(self, state=None):
        if state is None:
            state = jnp.array([0.0, 0.0, jnp.pi, 0.0])
        self._state = state
        return self._state

    def getState(self):
        return self._state

    def setState(self, state):
        self._state = jnp.asarray(state)

    @staticmethod
    @jit
    def _step(state, action, params):
        cart_pos, cart_vel, pole_angle, pole_vel = state

        s = jnp.sin(pole_angle)
        c = jnp.cos(pole_angle)
        m = 4.0 * (params["cart_mass"] + params["pole_mass"]) - 3.0 * params["pole_mass"] * (c**2)

        cart_accel = (
            2.0 * (params["pole_length"] * params["pole_mass"] * (pole_vel**2) * s + 2.0 * (action - params["mu_c"] * cart_vel))
            - 3.0 * params["pole_mass"] * params["gravity"] * c * s
            + 6.0 * params["mu_p"] * pole_vel * c / params["pole_length"]
        ) / m

        pole_accel = (
            -3.0 * c * (2.0 / params["pole_length"]) *
            (params["pole_length"] / 2.0 * params["pole_mass"] * (pole_vel**2) * s + action - params["mu_c"] * cart_vel)
            + 6.0 * (params["cart_mass"] + params["pole_mass"]) / (params["pole_mass"] * params["pole_length"]) *
            (params["pole_mass"] * params["gravity"] * s - 2.0 / params["pole_length"] * params["mu_p"] * pole_vel)
        ) / m

        dt = params["delta_time"] / params["sim_steps"]
        new_cart_vel = cart_vel + dt * cart_accel
        new_pole_vel = pole_vel + dt * pole_accel
        new_pole_angle = pole_angle + dt * new_pole_vel
        new_cart_pos = cart_pos + dt * new_cart_vel

        return jnp.array([new_cart_pos, new_cart_vel, new_pole_angle, new_pole_vel])

    def _integrate(self, state, action):
        def sim_step(s, _):
            return self._step(s, action, self.params), None

        new_state, _ = lax.scan(sim_step, state, xs=None, length=self.params["sim_steps"])
        return new_state

    def performAction(self, action):
        """Advance the environment by one control step and update internal state."""
        self._state = self._integrate(self._state, action)
        return self._state

    def rollout(self, initial_state, policy_fn, T, obs_noise_std=0.05):
        """Simulate T steps under a policy function."""
        if obs_noise_std <= 0.0:
            raise ValueError(f"obs_noise_std must be non-negative, got {obs_noise_std}.")
            
        def step_fn(state, _):
            action = policy_fn(state)
            new_state = self._integrate(state, action)
            return new_state, (new_state, action)

        _, (state_seq, action_seq) = lax.scan(step_fn, initial_state, None, length=T)
        full_states = jnp.vstack([initial_state[None, :], state_seq])

        # Observation noise
        noise = obs_noise_std * random.normal(random.PRNGKey(0), full_states.shape)
        full_states = full_states + noise
        return full_states, action_seq

    def loss(self, sig=None):
        return loss(self._state, sig=sig)

    def remap_angle(self):
        self._state = self._state.at[2].set(remap_angle(self._state[2]))

    def action_tanh(self, action):
        return self.params["max_force"] * jnp.tanh(action / self.params["max_force"])