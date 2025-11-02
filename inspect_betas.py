import numpy as np
import matplotlib.pyplot as plt
import math

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    """
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == "ours":
        T = num_diffusion_timesteps
        betas = np.zeros(T + 1, dtype=np.float64)
        betas[T] = 0.999

        for t in range(T - 1, -1, -1):
            betas[t] = (((betas[t+1]**(1/2)*3/2)*(1-betas[t+1]) + betas[t+1]**(3/2))*(2/3))**2
            if np.isnan(betas[t]) or np.isinf(betas[t]) or betas[t] < 0 or betas[t] > 1:
                raise ValueError(f"Custom schedule produced invalid beta[{t}] = {betas[t]}")

        # The diffusion model expects T betas, not T+1.
        # The last value betas[T] was just a starting point for the recurrence.
        # It's not clear from the original code if betas[0] or betas[1:] of length T should be returned.
        # Let's assume the array should be of length T. The original code returns T+1, which might be the bug.
        # The GaussianDiffusion class seems to take whatever is given, and `self.num_timesteps` is set to the length of betas.
        # However, the loop to calculate it goes from T-1 down to 0, which is T values.
        # The `betas` array has T+1 values, from index 0 to T.
        # So we should probably return `betas[:-1]` which has length T.
        return betas[:-1]
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def main():
    num_diffusion_timesteps = 4000
    
    betas_linear = get_named_beta_schedule("linear", num_diffusion_timesteps)
    betas_cosine = get_named_beta_schedule("cosine", num_diffusion_timesteps)
    betas_ours = get_named_beta_schedule("ours", num_diffusion_timesteps)
    
    timesteps = np.arange(num_diffusion_timesteps)
    
    plt.figure(figsize=(12, 8))
    plt.plot(timesteps, betas_linear, label="Linear")
    plt.plot(timesteps, betas_cosine, label="Cosine")
    plt.plot(timesteps, betas_ours, label="Ours")
    plt.xlabel("Timestep")
    plt.ylabel("Beta")
    plt.title("Beta Schedules Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("beta_schedules.png")
    print("Saved beta schedules plot to beta_schedules.png")
    
    # Print some values for closer inspection
    print("\n--- First 5 beta values ---")
    print(f"Linear: {betas_linear[:5]}")
    print(f"Cosine: {betas_cosine[:5]}")
    print(f"Ours:   {betas_ours[:5]}")

    print("\n--- Last 5 beta values ---")
    print(f"Linear: {betas_linear[-5:]}")
    print(f"Cosine: {betas_cosine[-5:]}")
    print(f"Ours:   {betas_ours[-5:]}")

    print("\n--- Final beta values ---")
    print(f"Linear final: {betas_linear[-1]}")
    print(f"Cosine final: {betas_cosine[-1]}")
    print(f"Ours final:   {betas_ours[-1]}")


if __name__ == "__main__":
    main()
