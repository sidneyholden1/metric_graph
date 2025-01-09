import numpy as np

def check_convergence(L, base_tolerance=0.01, min_tolerance=1e-8, mean_previous_graph_size=None):

    if len(L) < 3:
        return False
    
    mean_previous_sampling = np.mean(L[:-1])
    current_mean = np.mean(L)
    dif_mean_previous_sampling = np.abs((current_mean - mean_previous_sampling) / mean_previous_sampling)

    if mean_previous_graph_size == None:

        adaptive_tolerance = 0.001
    
        # Check if confidence interval is narrow enough
        sample_std = np.std(L, ddof=1)
        z = 1.96  
        ci_half_width = z * (sample_std / np.sqrt(len(L)))
        relative_ci_width = 2 * ci_half_width / current_mean
        if relative_ci_width > adaptive_tolerance:
            return False
        
    else:
        dif_mean_previous_graph_size = np.abs((current_mean - mean_previous_graph_size) / mean_previous_graph_size)
        adaptive_tolerance = np.max((dif_mean_previous_graph_size * base_tolerance, min_tolerance))

    print(f"\nCurrent dif = {dif_mean_previous_sampling} vs. adaptive tol = {adaptive_tolerance}\n")

    if dif_mean_previous_sampling > adaptive_tolerance:
        return False
    
    return True
    

def sequential_Newton(guesses, eigs, min_eigenvalue=1, max_eigenvalue=5):
    """Search for closeish convergence. Then search for full convergence. 
    Then clean solutions (get rid of neg, too large, nans)"""
    solutions = []
    for guess in guesses:
        solution = eigs(guess, solve_type="SVD iterate", printerval=np.inf, tol=1e-4, max_steps=20)
        solutions.append(solution)
        _, index = np.unique(np.round(solutions, 3), return_index=True)
        solutions = list(np.array(solutions)[index])
    full_solutions = eigs(solutions, solve_type="SVD iterate", printerval=10, tol=1e-13, max_steps=20)
    _, inds = np.unique(np.round(full_solutions, 8), return_index=True)

    full_solutions = full_solutions[inds]
    mask_too_small = full_solutions < min_eigenvalue
    mask_too_big = full_solutions > max_eigenvalue
    mask_isnan = np.isnan(full_solutions)
    mask_good = ~ (mask_too_small | mask_too_big | mask_isnan)
    full_solutions = full_solutions[mask_good]

    return full_solutions

def style_plot(fig, ax, aspect="equal", no_ticks=False, no_xticks=False, no_yticks=False):

    ticklabel_fontsize_major = 25

    # Remove white space
    fig.tight_layout()

    # Makes axes aspect equal
    if aspect:
        ax.set_aspect(aspect)

    # Make tick labels bold font
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(ticklabel_fontsize_major)

    # Make border thicker
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # xy_ticks
    major_tick_length = 7
    minor_tick_length = 4
    if no_ticks:
        major_xtick_length = 0
        major_ytick_length = 0
        minor_xtick_length = 0
        minor_ytick_length = 0
    elif no_xticks:
        major_xtick_length = 0
        major_ytick_length = major_tick_length
        minor_xtick_length = 0
        minor_ytick_length = minor_tick_length
    elif no_yticks:
        major_xtick_length = major_tick_length
        major_ytick_length = 0
        minor_xtick_length = minor_tick_length
        minor_ytick_length = 0
    else:
        major_xtick_length = major_tick_length
        major_ytick_length = major_tick_length
        minor_xtick_length = minor_tick_length
        minor_ytick_length = minor_tick_length
    ax.tick_params(axis='x', which='major', length=major_xtick_length, width=2, direction='out')
    ax.tick_params(axis='y', which='major', length=major_ytick_length, width=2, direction='out')
    ax.tick_params(axis='x', which='minor', length=minor_xtick_length, width=1, direction='out')
    ax.tick_params(axis='y', which='minor', length=minor_ytick_length, width=1, direction='out')

    for label in ax.get_xticklabels(minor=True) + ax.get_yticklabels(minor=True):
        label.set_fontweight('bold')
        label.set_fontsize(0.55 * ticklabel_fontsize_major)