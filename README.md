# Inverted Pendulum Controller via Machine Learning

This project develops a machine learning approach to model and control an inverted pendulum (cart-pole) system. The goal is to 1) learn the system dynamics from simulated data and 2) design a policy that stabilizes the pendulum in its upright position.

## Project Overview
The system state is defined by `[cart position, cart velocity, pole angle, pole angular velocity]`, and the control action is a horizontal force applied to the cart. The objective is to maintain the pole at the unstable upright equilibrium (`pole angle = 0`).
<div align="center">
  <img width="224" height="196" alt="Cart-pole system" 
       src="https://github.com/user-attachments/assets/afa1c01f-7b73-44f6-a2b9-b756024d12c1" />
  <p><em>Figure 1:</em> The inverted pendulum (cart-pole) system setup.</p>
</div>

## Results
Both model-free and model-based controllers were able to stabilize the pendulum near the upright equilibrium. The nonlinear model achieved more accurate long-horizon predictions and enabled successful policy optimization even under moderate observation noise.
<div align="center">
  <img width="504" height="276" alt="Model-free controller results" src="https://github.com/user-attachments/assets/02132f3b-758d-441b-8838-aff5c3eb61e6" />
  <p><em>Figure 2:</em> Results of the model-free controller.</p>
  <img width="511" height="271" alt="Model-based controller results" src="https://github.com/user-attachments/assets/207d21ac-e9ce-4317-bd40-bae8542d3d8e" />
  <p><em>Figure 3:</em> Results of the model-based controller.</p>
</div>

## Methodology
The project has several parts:

1. **System Simulation and Linear Modelling:**  
   Using a Python simulation, we collected state–transition data and fitted a *linear model* of the dynamics.

2. **Nonlinear Modelling and Feature Engineering:**  
   To improve prediction accuracy, a *kernel-based regression model* with radial basis functions (RBF) was implemented. Hyperparameters such as length scales and regularization were tuned using *gradient-based optimization* with the **JAX** library for automatic differentiation. The angular state was represented using `sin(θ)` and `cos(θ)` to handle periodicity and improve model generalization.

3. **Control Policy Optimization:**  
   A *linear control policy* was optimized using the learned model (model-based policy optimization) and directly on the true system dynamics (model-free control). The policy parameters were optimized via gradient-free and gradient-based methods to minimize a trajectory loss that penalized deviation from the upright position.

4. **Noise and Robustness Analysis:**  
   Observation and process noise were introduced to study model degradation and policy robustness.

## Conclusion
Through this project, machine learning methods were applied to both *system modelling* and *control optimization*. Gradient-based learning and kernel models enabled accurate modelling of nonlinear dynamics and robust control performance.

