"""
Category 2: Degree Distribution (M7-M15).
"""

import logging
import numpy as np
from scipy.stats import skew, kurtosis, pearsonr, spearmanr
import powerlaw

logger = logging.getLogger(__name__)


def compute(dataset, G, G_undirected, triples, num_entities, num_relations, **shared):
    """Compute M7-M15 degree distribution metrics."""
    logger.info("Computing Category 2: Degree Distribution (M7-M15)")

    in_degrees = shared["in_degrees"]
    out_degrees = shared["out_degrees"]
    total_degrees = shared["total_degrees"]

    results = {}

    # M7: Average Degree
    results["M7_avg_in_degree"] = float(np.mean(in_degrees))
    results["M7_avg_out_degree"] = float(np.mean(out_degrees))
    results["M7_avg_total_degree"] = float(np.mean(total_degrees))

    # M8: Median Degree
    results["M8_median_in_degree"] = float(np.median(in_degrees))
    results["M8_median_out_degree"] = float(np.median(out_degrees))
    results["M8_median_total_degree"] = float(np.median(total_degrees))

    # M9: Degree Standard Deviation
    results["M9_std_total_degree"] = float(np.std(total_degrees, ddof=0))

    # M10: Degree Skewness (Fisher-Pearson, bias-corrected)
    results["M10_degree_skewness"] = float(skew(total_degrees, bias=False))

    # M11: Degree Kurtosis (excess, bias-corrected)
    results["M11_degree_kurtosis"] = float(kurtosis(total_degrees, fisher=True, bias=False))

    # M12: Min / Max Degree
    results["M12_min_degree"] = int(np.min(total_degrees))
    results["M12_max_degree"] = int(np.max(total_degrees))

    # M13: Proportion of Low-Degree Entities
    for k in [1, 2, 3, 5, 10]:
        results[f"M13_low_degree_leq_{k}"] = float(np.mean(total_degrees <= k))

    # M14: Power-Law Exponent
    nonzero_degrees = total_degrees[total_degrees > 0]
    if len(nonzero_degrees) > 0:
        fit = powerlaw.Fit(nonzero_degrees, discrete=True, verbose=False)
        results["M14_alpha"] = float(fit.alpha)
        results["M14_xmin"] = float(fit.xmin)
        results["M14_sigma"] = float(fit.sigma)

        for alt in ["lognormal", "exponential", "stretched_exponential"]:
            try:
                R, p = fit.distribution_compare("power_law", alt)
                results[f"M14_R_vs_{alt}"] = float(R)
                results[f"M14_p_vs_{alt}"] = float(p)
            except Exception as e:
                logger.warning("Power-law comparison vs %s failed: %s", alt, e)
                results[f"M14_R_vs_{alt}"] = None
                results[f"M14_p_vs_{alt}"] = None
    else:
        results["M14_alpha"] = None
        results["M14_xmin"] = None
        results["M14_sigma"] = None

    # M15: In-Degree / Out-Degree Correlation
    p_corr, p_pval = pearsonr(in_degrees, out_degrees)
    s_corr, s_pval = spearmanr(in_degrees, out_degrees)
    results["M15_pearson_corr"] = float(p_corr)
    results["M15_pearson_pval"] = float(p_pval)
    results["M15_spearman_corr"] = float(s_corr)
    results["M15_spearman_pval"] = float(s_pval)

    return results
