import matplotlib.pyplot as plt
import numpy as np

def omega_evolve(omegas, I_nn, s, filename='omega_evolution.png', 
                 title=r'$\omega$ Evolution',
                 y_limits=(1e-6, 1e-2, 6),
                 legend_loc='upper right',
                 save=True):
    # Create figure with subplots
    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": [
             'Ubuntu', # 'DejaVu Sans', 'Helvetica', 'Arial', 'Liberation Sans',
        ],
        "font.monospace": "Ubuntu Mono",
        "axes.labelweight": "bold",
    })

    # Omega evolution over lambda iterations
    for mode in range(s):
        if mode in I_nn:
            l1, = ax.plot(
                np.abs(omegas[mode, :]), linewidth=2, 
                color='orange', label='selected'
            )
        else:
            l2, = ax.plot(
                np.abs(omegas[mode, :]), linewidth=1, alpha=0.5, 
                color='darkblue', linestyle='--',
                label='not selected'
            )

    ax.set_xlabel(r'$\lambda$ iteration', fontsize=25)
    ax.set_ylabel(r'$\omega$ values', fontsize=25)
    ax.set_title(title, fontsize=30)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(y_limits[0], y_limits[1])
    ax.set_yticks(
        np.logspace(
            np.log10(y_limits[0]), np.log10(y_limits[1]), num=y_limits[2]
        )
    )
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(handles=[l1, l2], fontsize=23, loc=legend_loc)

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
