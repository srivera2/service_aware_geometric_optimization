@workspace /code

I am pivoting my existing research project to focus on "Robust Geometric Optimization of Single-Cell 5G/6G Deployments." 

My current stack uses a PyTorch optimization loop driving a differentiable ray-tracing engine built on Mitsuba 3/Dr.Jit. The gradients are currently flowing correctly and the existing code should be used as-is with little modifications to the handoffs between Dr.Jit and Pytorch.

Analyze my existing repository and help me refactor the pipeline to achieve the following 4 specific research objectives:

### 1. Scene & Blockage Batching (The "Robustness" Layer)
* **Current State:** The code likely renders a single static scene.
* **Requirement:** Refactor the scene generation to support a "Batch" of $N$ scenarios (e.g., N=5). 
    * Scenario 0: The empty static scene.
    * Scenarios 1-N: The same scene but with random conductive boxes (representing buses/trucks) instantiated at random positions along the street layer.
* **Goal:** The forward pass must return signal data for *all* $N$ scenarios simultaneously (or in a loop) to calculate a "worst-case" loss.

### 2. Transmitter Constraint Manifold
* **Current State:** The transmitter (Tx) placement can only be constrained to the roof polygon of the chosen building. The building information is extracted from the scene's .xml file.
* **Requirement:** Implement a differentiable "Manifold Constraint" for placement.
    * Define a fixed line segment representing a building roof edge (e.g., `p_start` to `p_end`).
    * Create a learnable scalar parameter `alpha` (sigmoid-activated, range 0-1).
    * Define Tx Position as: $P_{tx} = p_{start} + \alpha \cdot (p_{end} - p_{start})$.
    * Keep Orientation (Azimuth/Tilt) as standard learnable parameters. This is modeled as the "look_at" position in Sionna RT.

### 3. Mask-Based Targeting & Stochastic Sampling
* **Current State:** Using a user-defined target zone (dB) to model simple distributions for general area coverage.
* **Requirement:** Implement a "User-Defined Mask" system.
    * Allow import of a binary mask (e.g., a "Snake" path winding through the street).
    * **Optimization Efficiency:** In the training loop, do *not* render the full resolution. Implement a sampler that selects a random subset (e.g., 200 pixels) *inside* the mask and a subset *outside* the mask (for interference minimization) per iteration.

### 4. Robust Loss Function (SoftMin)
* **Current State:** Likely MSE, Huber or Cross Entropy used for convex optimization.
* **Requirement:** Implement a `robust_coverage_loss` in PyTorch.
    * Input: A tensor of SNRs of shape `[Batch_Size, Num_Sampled_Pixels]`.
    * Logic: Compute the `SoftMin` (LogSumExp) across the `Batch_Size` dimension first (finding the worst-case scenario for each pixel), then average across the `Num_Sampled_Pixels`.
    * Goal: Maximize the utility of the *worst* blockage snapshot.

**Action Plan:**
1.  Review the file handling the scene geometry.
2.  Review the file handling the optimization loop/loss.
3.  Propose the specific code changes to inject the "Blockage Batching" and the "Roof Edge Constraint."
4.  Work through the code changes systematically while maintaining comparability to research baselines including naive placement/orientation, stochastic modeling and empirical equations.