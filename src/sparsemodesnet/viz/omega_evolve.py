import matplotlib.pyplot as plt
import numpy as np

def omega_evolve(omegas, I_nn, s, filename='omega_evolution.png', save=True):
    # Create figure with subplots
    fig, ax = plt.subplots(1, 1, figsize=(11, 6))

    # Omega evolution over lambda iterations
    for mode in range(s):
        if mode in I_nn:
            ax.plot(np.abs(omegas[mode, :]), linewidth=2, 
                     label=f'Mode {mode+1}', color='orange')
        else:
            ax.plot(np.abs(omegas[mode, :]), linewidth=1, alpha=0.5, 
                     color='darkblue', linestyle='--')

    ax.set_xlabel('Lambda iteration', fontsize=16)
    ax.set_ylabel('Omega values', fontsize=16)
    ax.set_title('Evolution of Omega Values During Training', fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=14)
    # ax.legend(fontsize=14)

    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=200)
    plt.show()

    # Final omega values
    final_omegas = np.abs(omegas[:, -1])

    # Print statistics about omega values
    print(f"\nOmega Statistics:")
    print(f"Number of lambda iterations: {omegas.shape[1]}")
    print(f"Max final omega: {np.max(final_omegas):.6e}")
    print(f"Min final omega: {np.min(final_omegas[final_omegas > 1e-15]):.6e}")
