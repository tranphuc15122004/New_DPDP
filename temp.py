import math
import numpy as np
from algorithm.Test_algorithm.adaptive_ratio import (
    AdaptiveRatioParams,
    compute_adaptive_ratio,
    compute_adaptive_ratio_quadratic,
    compute_adaptive_ratio_concave,
)


# Global placeholder (mirrors structure in main code)
CROSSOVER_TYPE_RATIO = 0.0

def adaptive_local_configs(num_order: int, num_vehicles: int):
    global CROSSOVER_TYPE_RATIO
    # Standalone plotting parameters (can adjust here for experiments)
    params = AdaptiveRatioParams(
        threshold_orders=150,
        kww_beta=0.7,
        kww_tau_factor=10.0,
        min_ratio=0.0,
        max_ratio=1.0,
        vehicle_influence=0.0,
        pivot_fraction=0.5,
        logistic_slope=10,
        early_shape=0.7,
    )
    info = compute_adaptive_ratio(num_orders=num_order, num_vehicles=num_vehicles, p=params)
    CROSSOVER_TYPE_RATIO = info['ratio']
    # Map keys to match previous plotting usage
    info['applied_ratio'] = info['ratio']
    info['raw_base'] = info['base']
    return info


def adaptive_local_configs_quadratic(num_order: int, num_vehicles: int, power: float = 4.0):
    """Exponential variant (was quadratic) using normalized exp schedule.

    power (k) càng lớn -> giữ cao lâu hơn rồi rơi nhanh hơn về cuối (độ cong tăng).
    """
    global CROSSOVER_TYPE_RATIO
    params = AdaptiveRatioParams(
        threshold_orders=100,
        kww_beta=0.8,
        kww_tau_factor=10.0,
        min_ratio=0.0,
        max_ratio=1.0,
        vehicle_influence=0.0,
        pivot_fraction=0.5,
        logistic_slope=12.5,
        early_shape=0.7,
    )
    info = compute_adaptive_ratio_quadratic(num_orders=num_order, num_vehicles=num_vehicles, p=params, power=power, cutoff=True)
    CROSSOVER_TYPE_RATIO = info['ratio']
    info['applied_ratio'] = info['ratio']
    info['raw_base'] = info['base']
    return info


import matplotlib.pyplot as plt

def plot_adaptive_local_configs(
    num_vehicles_list=(5, 10, 20),
    order_min=0,
    order_max=200,
    show_threshold=True,
    save_path=None,
    show_quadratic=False,
    show_concave=False,
    exp_powers=(2.0, 4.0, 6.0),
):
    orders = np.arange(order_min, order_max + 1)
    plt.figure(figsize=(9, 5.2))
    threshold_T = None

    for nv in num_vehicles_list:
        ratios = []
        for num_order in orders:
            result = adaptive_local_configs(num_order, nv)
            ratios.append(result['applied_ratio'])
            if threshold_T is None:
                threshold_T = result['threshold_T']
        plt.plot(orders, ratios, label=f'Vehicles={nv}')
        if show_quadratic:
            for pw in exp_powers:
                ratios_q = [adaptive_local_configs_quadratic(o, nv, power=pw)['applied_ratio'] for o in orders]
                plt.plot(orders, ratios_q, linestyle='--', alpha=0.55, label=f'Exp(k={pw}) V={nv}')

    if show_threshold and threshold_T is not None:
        plt.axvline(threshold_T, color='red', linestyle='--', alpha=0.7, label=f'Threshold T={threshold_T}')

    plt.xlabel('Number of Orders')
    plt.ylabel('CROSSOVER_TYPE_RATIO')
    plt.title('Adaptive CROSSOVER_TYPE_RATIO (KWW decay to 0 at threshold)')
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    # Ví dụ: vẽ cho nhiều quy mô đội xe
    plot_adaptive_local_configs(show_quadratic=True, show_concave=True)