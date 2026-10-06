# Graph Structural Properties: Computation Specification (v2)

## Document Purpose

This document provides exhaustive instructions for computing structural properties of Knowledge Graph (KG) benchmark datasets. These metrics will be used in an empirical study analysing the effectiveness of entity-agnostic KG embedding methods. The computed properties will be correlated with model performance (MRR, Hits@K) to explain when and why entity-agnostic approaches succeed or fail.

**All metrics must be computed for each dataset independently.** The final output should be a set of CSV/JSON files containing all metric values, plus any distribution-level data (histograms, frequency tables) saved separately.

---

## Academic References for Metrics

The following references underpin the metrics used in this specification. When implementing, **always prefer established library functions over hand-rolled implementations** so that metric validity is beyond dispute during peer review.

| Reference Key | Citation | Used For |
|---|---|---|
| [Shannon1948] | Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423. | All entropy metrics (M17, M18, M29, M57–M60) |
| [Gini1912] | Gini, C. (1912). Variabilità e mutabilità. *Reprinted in Memorie di metodologica statistica* (Ed. Pizetti & Salvemini, 1955). | Gini coefficient (M19) |
| [Clauset2009] | Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-law distributions in empirical data. *SIAM Review*, 51(4), 661–703. | Power-law fitting (M14) |
| [Newman2002] | Newman, M. E. J. (2002). Assortative mixing in networks. *Physical Review Letters*, 89(20), 208701. | Degree assortativity (M46) |
| [Newman2003] | Newman, M. E. J. (2003). The structure and function of complex networks. *SIAM Review*, 45(2), 167–256. | Clustering coefficient (M43, M44), connected components (M37–M40), diameter (M41) |
| [WattsStrogatz1998] | Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. *Nature*, 393(6684), 440–442. | Local clustering coefficient definition (M43) |
| [BarabasiAlbert1999] | Barabási, A.-L., & Albert, R. (1999). Emergence of scaling in random networks. *Science*, 286(5439), 509–512. | Power-law degree distributions in networks (M14 context) |
| [Garlaschelli2004] | Garlaschelli, D., & Loffredo, M. I. (2004). Patterns of link reciprocity in directed networks. *Physical Review Letters*, 93(26), 268701. | Reciprocity (M45) |
| [SunYuHan2011] | Sun, Y., & Han, J. (2012). Mining heterogeneous information networks: a structural analysis approach. *ACM SIGKDD Explorations Newsletter*, 14(2), 20–28. | Meta-paths (M62) |
| [Pearson1895] | Pearson, K. (1895). Notes on regression and inheritance in the case of two parents. *Proceedings of the Royal Society of London*, 58, 240–242. | Pearson correlation (M15) |
| [Spearman1904] | Spearman, C. (1904). The proof and measurement of association between two things. *The American Journal of Psychology*, 15(1), 72–101. | Spearman rank correlation (M15) |
| [Fisher1930] | Fisher, R. A. (1930). The moments of the distribution for normal samples of measures of departure from normality. *Proceedings of the Royal Society of London A*, 130(812), 16–28. | Skewness and kurtosis (M10, M11) |
| [Cover2006] | Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley. | Conditional entropy (M58–M60) |
| [Teru2020] | Teru, K. K., Denis, E. G., & Hamilton, W. L. (2020). Inductive relation prediction by subgraph reasoning. In *ICML*. | Inductive splits (M52–M56) |
| [Galkin2022] | Galkin, M., Denis, E. G., Wu, J., & Hamilton, W. L. (2022). NodePiece: Compositional and parameter-efficient representations of large knowledge graphs. In *ICLR*. | Relational context uniqueness concept (M24–M30) |

### Required Libraries and Their Roles

```
pip install pykeen networkx numpy scipy pandas matplotlib seaborn powerlaw graph-tool
```

| Library | Version | Used For | Why |
|---|---|---|---|
| `scipy.stats.entropy` | ≥1.10 | All Shannon entropy calculations | Peer-reviewed, handles edge cases (zero probabilities) correctly |
| `scipy.stats.skew` | ≥1.10 | Fisher's skewness (M10) | Standard implementation with bias correction option |
| `scipy.stats.kurtosis` | ≥1.10 | Excess kurtosis (M11) | Fisher's definition by default |
| `scipy.stats.pearsonr` | ≥1.10 | Pearson correlation + p-value (M15) | Returns both statistic and significance |
| `scipy.stats.spearmanr` | ≥1.10 | Spearman correlation + p-value (M15) | Returns both statistic and significance |
| `powerlaw` | ≥1.5 | Power-law fitting (M14) | Implements Clauset et al. (2009) exactly |
| `networkx` | ≥3.0 | All graph-level metrics (M37–M46) | Standard graph analysis library |
| `numpy` | ≥1.24 | Array operations, bincount, descriptive stats | Foundation library |
| `pandas` | ≥2.0 | Data storage and CSV output | Standard data handling |

**CRITICAL RULE**: For any metric where a `scipy`, `networkx`, or `powerlaw` library function exists, **use the library function**. Do NOT re-implement entropy, Gini, correlation, clustering, etc. from scratch. If a hand-rolled computation is unavoidable (e.g., context uniqueness, reciprocity), verify it with a unit test on a small synthetic graph with known properties.

---

## Environment and Data Loading

### Dataset Loading

All datasets are loaded from the PyKEEN library. The four transductive benchmark datasets are:

```python
from pykeen.datasets import FB15k237, WN18RR, CoDExLarge, YAGO310

datasets = {
    "FB15k-237": FB15k237(),
    "WN18RR": WN18RR(),
    "CoDEx-Large": CoDExLarge(),
    "YAGO3-10": YAGO310(),
}
```

For the **inductive link prediction** benchmarks (Teru splits) [Teru2020], load the FB15k-237 and WN18RR inductive versions (V1–V4). These are available via PyKEEN or can be loaded from the original Teru et al. data. Each version has a **training graph** and a separate **inference graph** (used for validation and testing). Metrics should be computed for **both** the training and inference graphs of each version.

```python
# Example for Teru inductive splits (check PyKEEN availability)
# If not directly in PyKEEN, download from:
# https://github.com/kkteru/grail/tree/master/data
# Each split contains: train.txt, valid.txt, test.txt for training graph
# and inductive versions for the inference graph
```

### Graph Construction

For each dataset, construct a **directed multigraph** from the **training triples only** (not validation/test). Each triple (h, r, t) becomes a directed edge from h to t with label r.

```python
import networkx as nx

def build_graph(triples_factory):
    """
    Build a NetworkX directed multigraph from a PyKEEN TriplesFactory.
    
    Args:
        triples_factory: PyKEEN TriplesFactory (use dataset.training)
    
    Returns:
        G: nx.MultiDiGraph with edge attribute 'relation' (integer ID)
        triples: numpy array of shape (num_triples, 3) with columns [head, relation, tail]
    """
    G = nx.MultiDiGraph()
    triples = triples_factory.mapped_triples.numpy()  # shape: (N, 3), columns: [h, r, t]
    
    # Add all entities as nodes (some may be isolates in training)
    num_entities = triples_factory.num_entities
    G.add_nodes_from(range(num_entities))
    
    # Add edges
    for h, r, t in triples:
        G.add_edge(int(h), int(t), relation=int(r))
    
    return G, triples
```

**Important**: Also construct an **undirected simple graph** (ignoring edge direction and relation types) for connectivity metrics:

```python
def build_undirected_graph(triples_factory):
    """Build undirected simple graph (no self-loops, no parallel edges)."""
    G_undirected = nx.Graph()
    triples = triples_factory.mapped_triples.numpy()
    num_entities = triples_factory.num_entities
    G_undirected.add_nodes_from(range(num_entities))
    for h, r, t in triples:
        G_undirected.add_edge(int(h), int(t))
    return G_undirected
```

---

## Path Mining (Replicating the Original Paper)

The original paper mines paths as follows. This procedure must be replicated exactly for the path-related metrics.

### CRITICAL: Path Type Clarification

The original paper uses **entity-relation paths** (NOT purely relational paths). This distinction is essential:

- **Entity-relation path**: A sequence alternating between entities and relations:
  `[e₀, r₀, e₁, r₁, e₂, r₂, ..., eₙ]`
  where each eᵢ ∈ E (entity) and each rᵢ ∈ R (relation), and (eᵢ, rᵢ, eᵢ₊₁) ∈ T for outgoing edges.

- **Relational path** (NOT what the paper uses): A sequence of only relation types: `[r₀, r₁, r₂, ...]`, which is sometimes called a "meta-path" [SunYuHan2011].

The paper explicitly states (Section 3.4, lines 213–215): *"each path consists of nodes and edges which are projected into a continuous space using the node projector and an embedding layer for relations, respectively."* The Transformer encoder processes the **full interleaved sequence** of entity projections and relation embeddings.

**Therefore**: When mining paths, the output MUST be the full entity-relation sequence `[e, r, e, r, ..., e]`, NOT just the relation sequence. Both entity IDs and relation IDs must be preserved. Entities in the path are later projected via the node projector (using their relational context), and relations are looked up from the embedding table.

### Path Mining Algorithm

For each entity `e` in the KG:
1. Attempt to mine `N` unique **entity-relation** paths (where N is a configurable parameter; in experiments N ∈ {1, 2, 4, 8} paths per entity).
2. Each path is either **outgoing** (starting from `e`) or **incoming** (ending at `e`), chosen with equal probability (50/50).
3. Paths are generated via **random walks** on the KG.
4. Paths must be **loop-free**: no entity node in the path may appear more than once.
5. Path length is fixed (in experiments, maximum length = 20 **nodes**, meaning up to 20 entities and 19 relations in the sequence).

### Implementation Specification

```python
import random
import numpy as np

def mine_paths_for_entity(G, entity, num_paths, max_length=20, max_attempts_multiplier=10):
    """
    Mine unique loop-free random walk entity-relation paths for a single entity.
    
    IMPORTANT: Paths are ENTITY-RELATION paths, NOT relational paths.
    Each path is a list: [entity_0, relation_0, entity_1, relation_1, ..., entity_n]
    where entity_i are entity integer IDs and relation_i are relation integer IDs.
    The path represents a walk through the KG where each consecutive 
    (entity_i, relation_i, entity_{i+1}) is a triple in the training set.
    
    Args:
        G: nx.MultiDiGraph — the knowledge graph
        entity: int — the entity ID to mine paths for
        num_paths: int — number of unique paths to attempt to mine
        max_length: int — maximum path length in NODES/ENTITIES (default: 20).
                         A path of max_length=20 contains up to 20 entity IDs and 
                         19 relation IDs, for a total sequence length of 39 elements.
        max_attempts_multiplier: int — multiplier for max attempts
    
    Returns:
        paths: list of lists, where each path is a list of alternating 
               [entity, relation, entity, relation, ..., entity] elements.
               - For outgoing paths, paths[i][0] == entity (path starts at entity).
               - For incoming paths, paths[i][-1] == entity (path ends at entity).
               - Odd-indexed elements (1, 3, 5, ...) are RELATION IDs.
               - Even-indexed elements (0, 2, 4, ...) are ENTITY IDs.
        directions: list of str, either "outgoing" or "incoming" for each path
    
    Path structure example (outgoing, length 4 nodes):
        [entity, rel_0, e_1, rel_1, e_2, rel_2, e_3]
         ^even   ^odd   ^even ^odd  ^even ^odd   ^even
    
    Path structure example (incoming, length 3 nodes):
        [e_2, rel_1, e_1, rel_0, entity]
        Walk was: entity <- e_1 <- e_2 (following reverse edges), 
        then reversed so entity is at the end.
    """
    paths = []
    directions = []
    seen_paths = set()
    max_attempts = num_paths * max_attempts_multiplier
    
    for _ in range(max_attempts):
        if len(paths) >= num_paths:
            break
        
        direction = random.choice(["outgoing", "incoming"])
        path = _random_walk(G, entity, max_length, direction)
        
        if path is not None:
            # Uniqueness is determined by the FULL entity-relation sequence
            path_key = tuple(path)
            if path_key not in seen_paths:
                seen_paths.add(path_key)
                paths.append(path)
                directions.append(direction)
    
    return paths, directions


def _random_walk(G, start_entity, max_length, direction):
    """
    Perform a single loop-free random walk, producing an entity-relation path.
    
    Args:
        G: nx.MultiDiGraph
        start_entity: int — starting node
        max_length: int — maximum number of ENTITY nodes in the path
        direction: str — "outgoing" or "incoming"
    
    Returns:
        path: list of [entity, relation, entity, relation, ..., entity] or None if walk fails.
              The path always contains entity IDs at even indices and relation IDs at odd indices.
    
    For "outgoing": follows forward edges. path[0] = start_entity.
    For "incoming": follows reverse edges from start_entity, then the resulting 
                    path is stored such that start_entity is the LAST entity.
    """
    visited = {start_entity}
    # Internal walk always goes "forward" from start; for incoming, we reverse at the end
    path = [start_entity]
    current = start_entity
    
    for step in range(max_length - 1):
        if direction == "outgoing":
            neighbors = []
            for _, neighbor, data in G.out_edges(current, data=True):
                if neighbor not in visited:
                    neighbors.append((neighbor, data['relation']))
        else:  # incoming — follow reverse edges
            neighbors = []
            for predecessor, _, data in G.in_edges(current, data=True):
                if predecessor not in visited:
                    neighbors.append((predecessor, data['relation']))
        
        if not neighbors:
            break
        
        next_node, relation = random.choice(neighbors)
        path.append(int(relation))  # Relation ID at odd index
        path.append(int(next_node))  # Entity ID at even index
        visited.add(next_node)
        current = next_node
    
    # Path must have at least 2 entity nodes (1 relation edge) to be useful
    # Minimum valid path: [entity_0, relation_0, entity_1] — length 3
    if len(path) < 3:
        return None
    
    # For incoming paths: reverse the sequence so start_entity is at the end
    if direction == "incoming":
        path = path[::-1]
        # After reversal: entities are still at even indices, relations at odd indices
        # because the path length is always odd (starts and ends with entity)
    
    return path
```

### Verification of Path Structure

After mining, verify the path structure with this assertion:

```python
def verify_path(path):
    """Verify that a mined path has correct entity-relation alternating structure."""
    assert len(path) >= 3, f"Path too short: {path}"
    assert len(path) % 2 == 1, f"Path must have odd length (entity-relation-...-entity): len={len(path)}"
    # Even indices should be entity IDs, odd indices should be relation IDs
    # (we can't verify types further without the graph, but structure must be correct)
    return True

# Run verification on all mined paths
for entity in entity_paths_dict:
    for path in entity_paths_dict[entity]:
        verify_path(path)
```

### Path Mining for Metrics

For the purpose of computing path-related metrics (Category 5), mine paths for **all entities** in each dataset using the following configuration:

```python
PATH_CONFIGS = {
    "num_paths_per_entity": 8,   # Use the maximum from the paper
    "max_length": 20,            # Max 20 entity nodes per path (same as paper)
    "random_seed": 42,           # For reproducibility
}
```

Store the mined entity-relation paths for analysis. Both the full entity-relation paths and derived relational paths (extracted by taking only odd-indexed elements) are used in subsequent metrics.

```python
def extract_relation_sequence(entity_relation_path):
    """
    Extract the relation-only sequence from an entity-relation path.
    
    Input:  [e0, r0, e1, r1, e2, r2, e3]  (entity-relation path)
    Output: (r0, r1, r2)                    (relation sequence / meta-path)
    """
    return tuple(entity_relation_path[i] for i in range(1, len(entity_relation_path), 2))

def extract_entity_sequence(entity_relation_path):
    """
    Extract the entity-only sequence from an entity-relation path.
    
    Input:  [e0, r0, e1, r1, e2, r2, e3]  (entity-relation path)
    Output: (e0, e1, e2, e3)               (entity sequence)
    """
    return tuple(entity_relation_path[i] for i in range(0, len(entity_relation_path), 2))
```

---

## Metric Definitions and Computation Instructions

Each metric below is described with:
- **ID**: Unique identifier (M1, M2, ...) for cross-referencing
- **Name**: Human-readable name
- **Category**: Grouping
- **Definition**: Precise mathematical or algorithmic definition
- **Reference**: Academic source for the metric or method
- **Library**: Mandatory library function to use (if applicable)
- **Rationale**: Why this metric matters for entity-agnostic methods
- **Computation**: Exact code or pseudocode
- **Output type**: Scalar, vector, distribution, etc.

---

### CATEGORY 1: Basic Graph Statistics

These provide foundational context for each KG. These are standard descriptive statistics of knowledge graphs as reported in the KGE literature (e.g., Bordes et al., 2013; Dettmers et al., 2018).

#### M1: Number of Entities

- **Definition**: |E| = total number of unique entity IDs in the dataset (across all splits: train + valid + test).
- **Rationale**: Determines the scale of the disambiguation problem. More entities with the same relation vocabulary means harder differentiation.
- **Computation**:
  ```python
  num_entities = dataset.num_entities  # PyKEEN provides this
  ```
- **Output**: Single integer per dataset.

#### M2: Number of Relations

- **Definition**: |R| = total number of unique relation types in the dataset.
- **Rationale**: This is the size of the learnable parameter space for the entity-agnostic framework. The framework stores embeddings only for relations, so |R| determines model capacity.
- **Computation**:
  ```python
  num_relations = dataset.num_relations
  ```
- **Output**: Single integer per dataset.

#### M3: Number of Triples (per split)

- **Definition**: Count of (h, r, t) triples in each split (train, validation, test) and total.
- **Rationale**: Total evidence available for learning. More triples per entity means more context.
- **Computation**:
  ```python
  num_train = dataset.training.num_triples
  num_valid = dataset.validation.num_triples
  num_test = dataset.testing.num_triples
  num_total = num_train + num_valid + num_test
  ```
- **Output**: Four integers per dataset.

#### M4: Graph Density

- **Definition**: The ratio of existing triples to the maximum possible number of triples.
  
  density = |T_train| / (|E| × |R| × |E|)
  
  Also compute simple edge density (ignoring relation types):
  
  simple_density = |unique (h,t) pairs in T_train| / (|E| × (|E| - 1))

- **Reference**: Standard graph density definition [Newman2003].
- **Rationale**: Denser graphs provide more relational context per entity. Very sparse graphs limit the framework's ability to construct discriminative representations.
- **Library**: Use `nx.density(G_undirected)` for the simple graph density as a cross-check.
- **Computation**:
  ```python
  triples = dataset.training.mapped_triples.numpy()
  num_e = dataset.num_entities
  num_r = dataset.num_relations
  
  # Full density (including relation dimension)
  full_density = len(triples) / (num_e * num_r * num_e)
  
  # Simple edge density (ignoring relation types) via NetworkX
  G_undirected = build_undirected_graph(dataset.training)
  simple_density = nx.density(G_undirected)
  # Note: nx.density computes 2*|edges| / (|V|*(|V|-1)) for undirected graphs
  ```
- **Output**: Two floats per dataset.

#### M5: Relation-to-Entity Ratio

- **Definition**: |R| / |E|
- **Rationale**: Measures how expressive the relational vocabulary is relative to the number of entities it must distinguish. A higher ratio means the framework has more "building blocks" per entity.
- **Computation**:
  ```python
  r_to_e_ratio = num_relations / num_entities
  ```
- **Output**: Single float per dataset.

#### M6: Triple-to-Entity Ratio

- **Definition**: |T_train| / |E|
- **Rationale**: Average amount of evidence (triples) per entity. Equivalent to the average total degree in the training graph.
- **Computation**:
  ```python
  t_to_e_ratio = num_train / num_entities
  ```
- **Output**: Single float per dataset.

---

### CATEGORY 2: Degree Distribution

Degree determines how much local context is available per entity. All degree metrics are computed on the **training graph only**.

#### M7: Average Degree (in, out, total)

- **Definition**: 
  - In-degree of entity e: |{(h, r, e) ∈ T_train}|
  - Out-degree of entity e: |{(e, r, t) ∈ T_train}|
  - Total degree: in-degree + out-degree
  - Report the mean of each across all entities.
- **Reference**: Standard graph theory [Newman2003].
- **Rationale**: The node projector uses the relational context (derived from the entity's neighbourhood). Higher degree = more context.
- **Computation**:
  ```python
  triples = dataset.training.mapped_triples.numpy()
  num_e = dataset.num_entities
  
  in_degrees = np.zeros(num_e)
  out_degrees = np.zeros(num_e)
  
  for h, r, t in triples:
      out_degrees[int(h)] += 1
      in_degrees[int(t)] += 1
  
  total_degrees = in_degrees + out_degrees
  
  avg_in_degree = np.mean(in_degrees)
  avg_out_degree = np.mean(out_degrees)
  avg_total_degree = np.mean(total_degrees)
  ```
- **Output**: Three floats per dataset. Also store the full degree arrays for subsequent metrics.

#### M8: Median Degree (in, out, total)

- **Definition**: Median of the in-degree, out-degree, and total degree distributions.
- **Rationale**: More robust to outliers (hubs) than the mean. If the median is much lower than the mean, most entities have low degree and the graph is hub-dominated.
- **Computation**:
  ```python
  median_in = np.median(in_degrees)
  median_out = np.median(out_degrees)
  median_total = np.median(total_degrees)
  ```
- **Output**: Three floats per dataset.

#### M9: Degree Standard Deviation

- **Definition**: Standard deviation of the total degree distribution.
- **Rationale**: High variance indicates heterogeneous connectivity — some entities have rich contexts while others are impoverished.
- **Computation**:
  ```python
  std_total_degree = np.std(total_degrees, ddof=0)  # population std
  ```
- **Output**: Single float per dataset.

#### M10: Degree Skewness

- **Definition**: Fisher's skewness coefficient of the total degree distribution [Fisher1930]:
  
  γ₁ = m₃ / m₂^(3/2)
  
  where m₂, m₃ are the second and third central moments.
  
- **Reference**: [Fisher1930]. Implemented in `scipy.stats.skew`.
- **Library**: **MUST use** `scipy.stats.skew(total_degrees, bias=False)` for the corrected Fisher–Pearson coefficient.
- **Rationale**: Positive skewness (long right tail) means most entities have low degree while a few have very high degree. High skewness is problematic for entity-agnostic methods.
- **Computation**:
  ```python
  from scipy.stats import skew
  degree_skewness = skew(total_degrees, bias=False)
  ```
- **Output**: Single float per dataset.

#### M11: Degree Kurtosis

- **Definition**: Excess kurtosis (Fisher's definition) of the total degree distribution [Fisher1930]:
  
  κ = m₄ / m₂² − 3
  
- **Reference**: [Fisher1930]. Implemented in `scipy.stats.kurtosis`.
- **Library**: **MUST use** `scipy.stats.kurtosis(total_degrees, fisher=True, bias=False)`.
- **Rationale**: High kurtosis indicates heavy tails — extreme hubs and/or extreme low-degree nodes.
- **Computation**:
  ```python
  from scipy.stats import kurtosis
  degree_kurtosis = kurtosis(total_degrees, fisher=True, bias=False)
  ```
- **Output**: Single float per dataset.

#### M12: Min / Max Degree

- **Definition**: Minimum and maximum total degree across all entities.
- **Rationale**: Min degree reveals the worst case — entities with the least context. If min degree is 0 or 1, some entities are essentially unrepresentable.
- **Computation**:
  ```python
  min_degree = int(np.min(total_degrees))
  max_degree = int(np.max(total_degrees))
  ```
- **Output**: Two integers per dataset.

#### M13: Proportion of Low-Degree Entities

- **Definition**: Fraction of entities with total degree ≤ k, for k ∈ {1, 2, 3, 5, 10}. This is the empirical CDF evaluated at each threshold.
- **Rationale**: These are the entities most likely to be indistinguishable under the entity-agnostic scheme.
- **Computation**:
  ```python
  thresholds = [1, 2, 3, 5, 10]
  low_degree_fractions = {}
  for k in thresholds:
      low_degree_fractions[k] = float(np.mean(total_degrees <= k))
  ```
- **Output**: Five floats per dataset (one per threshold).

#### M14: Power-Law Exponent

- **Definition**: The exponent α of a power-law fit P(k) ∝ k^(−α) to the total degree distribution, estimated via the maximum likelihood method of Clauset, Shalizi, & Newman (2009).
- **Reference**: [Clauset2009]. This is the gold-standard method for testing power-law hypotheses.
- **Library**: **MUST use** the `powerlaw` Python package (which implements [Clauset2009] exactly). Do NOT fit via linear regression on log-log plots (this is known to produce biased estimates; see Section 3.1 of [Clauset2009]).
- **Rationale**: Many real-world graphs follow power-law degree distributions [BarabasiAlbert1999]. The exponent characterises how quickly the distribution falls off.
- **Computation**:
  ```python
  import powerlaw
  
  # Filter out zero-degree entities (powerlaw library requires positive values)
  nonzero_degrees = total_degrees[total_degrees > 0]
  
  fit = powerlaw.Fit(nonzero_degrees, discrete=True, verbose=False)
  alpha = fit.alpha          # MLE power-law exponent
  xmin = fit.xmin            # Estimated lower bound for power-law behaviour
  sigma = fit.sigma          # Standard error of alpha
  
  # Hypothesis testing: compare power-law vs. alternative distributions
  # Positive R means power-law is preferred; p < 0.05 means significant
  R_lognormal, p_lognormal = fit.distribution_compare('power_law', 'lognormal')
  R_exponential, p_exponential = fit.distribution_compare('power_law', 'exponential')
  R_stretched_exp, p_stretched_exp = fit.distribution_compare('power_law', 'stretched_exponential')
  ```
- **Output**: α, σ, xmin (floats), comparison R-values and p-values. Report whether power-law is preferred.

#### M15: In-Degree / Out-Degree Correlation

- **Definition**: Pearson [Pearson1895] and Spearman [Spearman1904] correlation between in-degree and out-degree per entity.
- **Reference**: [Pearson1895], [Spearman1904].
- **Library**: **MUST use** `scipy.stats.pearsonr` and `scipy.stats.spearmanr`. These return the correlation coefficient AND two-sided p-value.
- **Rationale**: If entities that receive many edges also emit many, their contexts are richer in both directions.
- **Computation**:
  ```python
  from scipy.stats import pearsonr, spearmanr
  
  pearson_corr, pearson_p = pearsonr(in_degrees, out_degrees)
  spearman_corr, spearman_p = spearmanr(in_degrees, out_degrees)
  ```
- **Output**: Four floats per dataset (two correlations + two p-values).

---

### CATEGORY 3: Relational Diversity and Distribution

The framework's only learned parameters are relation embeddings, so the properties of the relation distribution directly determine model capacity.

#### M16: Relation Frequency Distribution

- **Definition**: For each relation type r ∈ R, count the number of triples in the training set that use r. Report the full distribution.
- **Rationale**: Skewed distributions mean some relations dominate, reducing the effective vocabulary.
- **Computation**:
  ```python
  triples = dataset.training.mapped_triples.numpy()
  relation_counts = np.bincount(triples[:, 1].astype(int), minlength=num_relations)
  
  import pandas as pd
  rel_freq_df = pd.DataFrame({
      'relation_id': range(num_relations),
      'count': relation_counts,
      'fraction': relation_counts / len(triples)
  }).sort_values('count', ascending=False).reset_index(drop=True)
  ```
- **Output**: Full table (|R| rows) saved as CSV. Also save a histogram plot.

#### M17: Relation Entropy

- **Definition**: Shannon entropy [Shannon1948] of the relation frequency distribution:
  
  H(R) = −Σᵣ p(r) log₂ p(r)
  
  where p(r) = count(r) / |T_train|.
  
- **Reference**: [Shannon1948].
- **Library**: **MUST use** `scipy.stats.entropy(pk, base=2)`. This function correctly handles the convention that 0 log 0 = 0.
- **Rationale**: Higher entropy = more uniform relation usage = more diverse signals for the framework.
- **Computation**:
  ```python
  from scipy.stats import entropy as scipy_entropy
  
  relation_probs = relation_counts / relation_counts.sum()
  relation_entropy = scipy_entropy(relation_probs, base=2)
  # scipy.stats.entropy automatically handles p=0 entries (treats 0*log(0) as 0)
  ```
- **Output**: Single float per dataset (in bits).

#### M18: Normalised Relation Entropy

- **Definition**: H(R) / H_max where H_max = log₂(|R|) is the entropy of a uniform distribution over |R| relations. This normalisation is sometimes called "evenness" or "efficiency" in ecology [Shannon1948].
  
  Normalised entropy ∈ [0, 1], where 1 means perfectly uniform.
  
- **Reference**: [Shannon1948]; normalisation is standard in ecology and information theory.
- **Computation**:
  ```python
  max_entropy = np.log2(num_relations)
  normalised_entropy = relation_entropy / max_entropy if max_entropy > 0 else 0
  ```
- **Output**: Single float per dataset in [0, 1].

#### M19: Relation Gini Coefficient

- **Definition**: Gini coefficient [Gini1912] of the relation frequency distribution. For n values sorted in non-decreasing order y₁ ≤ y₂ ≤ ... ≤ yₙ:
  
  G = (2 Σᵢ₌₁ⁿ i·yᵢ) / (n Σᵢ₌₁ⁿ yᵢ) − (n+1)/n
  
  This is equivalent to half the relative mean absolute difference: G = Σᵢ Σⱼ |yᵢ − yⱼ| / (2n² ȳ).
  
- **Reference**: [Gini1912]. The formula above is the standard unbiased estimator for discrete distributions.
- **Library**: While there is no single canonical library function, the following implementation matches the standard definition exactly. As a cross-check, verify that the Gini coefficient satisfies: G = 0 when all values are equal; G approaches 1 when one value dominates.
- **Rationale**: Gini = 0 means perfect equality, Gini → 1 means one relation dominates.
- **Computation**:
  ```python
  def gini_coefficient(values):
      """
      Compute Gini coefficient. Matches the standard definition in [Gini1912].
      Equivalent to: G = (2 * sum(i * y_i)) / (n * sum(y_i)) - (n+1)/n
      for sorted values y_1 <= y_2 <= ... <= y_n.
      
      Cross-check: also equal to half the relative mean absolute difference:
      G = sum_i sum_j |y_i - y_j| / (2 * n^2 * mean(y))
      """
      values = np.array(values, dtype=np.float64)
      if np.all(values == 0):
          return 0.0
      sorted_vals = np.sort(values)
      n = len(sorted_vals)
      index = np.arange(1, n + 1)
      gini = (2.0 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1.0) / n
      return float(gini)
  
  relation_gini = gini_coefficient(relation_counts)
  
  # Cross-check with mean absolute difference formulation:
  def gini_mad(values):
      values = np.array(values, dtype=np.float64)
      if values.mean() == 0:
          return 0.0
      n = len(values)
      mad = np.sum(np.abs(values[:, None] - values[None, :])) / (2 * n * n * values.mean())
      return float(mad)
  
  assert abs(relation_gini - gini_mad(relation_counts)) < 1e-10, "Gini cross-check failed"
  ```
- **Output**: Single float per dataset in [0, 1].

#### M20: Top-k Relation Concentration

- **Definition**: Cumulative fraction of triples accounted for by the top k most frequent relations, for k ∈ {1, 2, 3, 5, 10}. This is also known as the concentration ratio CR_k, used in industrial economics.
- **Rationale**: Directly measures concentration. If the top 2 relations account for 74% of triples (as in WN18RR), the framework has very limited signal diversity.
- **Computation**:
  ```python
  sorted_fractions = np.sort(relation_counts / relation_counts.sum())[::-1]
  cumulative = np.cumsum(sorted_fractions)
  
  concentration = {}
  for k in [1, 2, 3, 5, 10]:
      concentration[k] = float(cumulative[min(k, len(cumulative)) - 1])
  ```
- **Output**: Five floats per dataset.

#### M21: Number of Rare Relations

- **Definition**: Count of relations appearing in fewer than X% of triples, for X ∈ {0.1%, 0.5%, 1%}.
- **Rationale**: Rare relations may be poorly learned but could be highly discriminative.
- **Computation**:
  ```python
  total_triples = len(triples)
  rare_counts = {}
  for threshold_pct in [0.001, 0.005, 0.01]:
      threshold = total_triples * threshold_pct
      rare_counts[threshold_pct] = int(np.sum(relation_counts < threshold))
  ```
- **Output**: Three integers per dataset.

#### M22: Relation Type Diversity per Entity

- **Definition**: For each entity e, count the number of distinct relation types it participates in (as head or tail):
  
  div(e) = |{r : ∃t. (e, r, t) ∈ T} ∪ {r : ∃h. (h, r, e) ∈ T}|
  
  Report mean, median, std, min, max across all entities.
  
- **Rationale**: This is the "effective vocabulary size" per entity. Higher diversity per entity means the node projector can create more discriminative representations.
- **Computation**:
  ```python
  from collections import defaultdict
  
  entity_relations = defaultdict(set)
  for h, r, t in triples:
      entity_relations[int(h)].add(int(r))
      entity_relations[int(t)].add(int(r))
  
  rel_diversity = np.array([
      len(entity_relations.get(e, set())) for e in range(num_entities)
  ])
  
  avg_rel_diversity = float(np.mean(rel_diversity))
  median_rel_diversity = float(np.median(rel_diversity))
  std_rel_diversity = float(np.std(rel_diversity))
  min_rel_diversity = int(np.min(rel_diversity))
  max_rel_diversity = int(np.max(rel_diversity))
  ```
- **Output**: Five values per dataset. Also store the full distribution.

#### M23: Normalised Relation Type Diversity per Entity

- **Definition**: div(e) / |R| for each entity. Report the average.
- **Rationale**: What fraction of the full relation vocabulary does the average entity use?
- **Computation**:
  ```python
  normalised_diversity = rel_diversity / num_relations
  avg_normalised_diversity = float(np.mean(normalised_diversity))
  ```
- **Output**: Single float per dataset in [0, 1].

---

### CATEGORY 4: Relational Context Uniqueness

This directly measures how well entities can be distinguished by their relational context alone — the core assumption of the node projector. Inspired by the relational context analysis in [Galkin2022].

#### M24: Relational Context Definition

- **Definition**: The relational context of an entity e is a pair of vectors:
  
  - **Incoming context**: c_in(e) ∈ ℤ^|R| where c_in(e)[r] = |{h : (h, r, e) ∈ T}|
  - **Outgoing context**: c_out(e) ∈ ℤ^|R| where c_out(e)[r] = |{t : (e, r, t) ∈ T}|
  - **Full context**: c(e) = [c_in(e); c_out(e)] ∈ ℤ^(2|R|)
  
  This is exactly the input to the node projector (cf. the adjacency matrix A_er in Section 3.3 of the original paper).
  
- **Rationale**: The node projector is a deterministic MLP: f(c(e)). If c(e₁) = c(e₂), then f(c(e₁)) = f(c(e₂)). The uniqueness of contexts is thus an upper bound on the framework's ability to distinguish entities via the node projector alone. (Paths can further disambiguate.)
- **Computation**:
  ```python
  incoming_contexts = np.zeros((num_entities, num_relations), dtype=np.int32)
  outgoing_contexts = np.zeros((num_entities, num_relations), dtype=np.int32)
  
  for h, r, t in triples:
      outgoing_contexts[int(h), int(r)] += 1
      incoming_contexts[int(t), int(r)] += 1
  
  full_contexts = np.concatenate([incoming_contexts, outgoing_contexts], axis=1)
  ```
- **Output**: Intermediate data (numpy array of shape |E| × 2|R|). Not reported directly.

#### M25: Proportion of Entities with Unique Relational Contexts

- **Definition**: 
  (a) distinct_fraction = |{distinct c(e)}| / |E|
  (b) unique_fraction = |{e : no other entity shares c(e)}| / |E|
  
- **Reference**: Related to the context uniqueness analysis in [Galkin2022] and the original PathE paper appendix.
- **Rationale**: Arguably the single most predictive metric. The original paper reports: FB15k-237 = 92%, WN18RR = 8%, CoDEx-Large = 46.4%, YAGO3-10 = 19.8%.
- **Computation**:
  ```python
  from collections import Counter
  
  context_tuples = [tuple(row) for row in full_contexts]
  context_counts = Counter(context_tuples)
  
  num_distinct_contexts = len(context_counts)
  distinct_fraction = num_distinct_contexts / num_entities
  
  num_unique = sum(1 for count in context_counts.values() if count == 1)
  unique_fraction = num_unique / num_entities
  ```
- **Output**: Two floats per dataset.

#### M26: Context Compression Ratio

- **Definition**: |distinct contexts| / |E|. Same as distinct_fraction from M25.
- **Output**: Single float per dataset in (0, 1].

#### M27: Largest Equivalence Class Size

- **Definition**: max_c |{e : c(e) = c}|
- **Rationale**: Worst-case ambiguity. If the largest class has 10,000 entities, the node projector maps all 10,000 identically.
- **Computation**:
  ```python
  largest_class = max(context_counts.values())
  ```
- **Output**: Single integer per dataset.

#### M28: Average Equivalence Class Size

- **Definition**: |E| / |distinct contexts|, equivalently the mean of all class sizes.
- **Computation**:
  ```python
  avg_class_size = num_entities / num_distinct_contexts
  ```
- **Output**: Single float per dataset.

#### M29: Entropy of Equivalence Class Size Distribution

- **Definition**: Shannon entropy [Shannon1948] of the probability distribution where each entity is assigned to its equivalence class. Let sᵢ be the size of class i. Then p(i) = sᵢ / |E|.
  
  H = −Σᵢ p(i) log₂ p(i)
  
  Maximum value is log₂(|E|) when every entity is unique.
  
- **Reference**: [Shannon1948], [Cover2006].
- **Library**: **MUST use** `scipy.stats.entropy`.
- **Computation**:
  ```python
  class_sizes = np.array(list(context_counts.values()))
  class_probs = class_sizes / class_sizes.sum()
  class_entropy = float(scipy_entropy(class_probs, base=2))
  ```
- **Output**: Single float per dataset.

#### M30: Proportion of Entities in the Largest Equivalence Class

- **Definition**: largest_class / |E|
- **Rationale**: Hard ceiling: if 50% of entities share one context, the node projector cannot disambiguate half the entities.
- **Computation**:
  ```python
  largest_class_fraction = largest_class / num_entities
  ```
- **Output**: Single float per dataset.

---

### CATEGORY 5: Path Availability and Richness

The framework relies on random walks to produce entity-relation paths. These metrics capture whether such paths exist and whether they carry diverse information. **Use the path mining procedure described above.**

**IMPORTANT**: All path metrics operate on entity-relation paths. When computing relation-level statistics from paths, extract the relation sequence using `extract_relation_sequence()`. When computing entity-level statistics, use `extract_entity_sequence()`.

#### M31: Average Number of Unique Entity-Relation Paths per Entity

- **Definition**: After attempting to mine N=8 entity-relation paths per entity (max_length=20 nodes), compute the average number of paths actually found.
- **Rationale**: If many entities cannot produce the requested number of unique paths, the framework has insufficient evidence.
- **Computation**:
  ```python
  paths_found = []
  entity_paths_dict = {}
  for entity in range(num_entities):
      entity_paths, directions = mine_paths_for_entity(G, entity, num_paths=8, max_length=20)
      entity_paths_dict[entity] = entity_paths
      paths_found.append(len(entity_paths))
  
  paths_found = np.array(paths_found)
  avg_paths = float(np.mean(paths_found))
  median_paths = float(np.median(paths_found))
  std_paths = float(np.std(paths_found))
  ```
- **Output**: Three floats per dataset. Also store the full distribution.

#### M32: Proportion of Entities with at Least k Unique Paths

- **Definition**: For k ∈ {1, 2, 4, 8}: fraction of entities for which at least k unique loop-free entity-relation paths were found.
- **Rationale**: Directly measures "path coverage". Thresholds match experimental settings in the original paper.
- **Computation**:
  ```python
  path_coverage = {}
  for k in [1, 2, 4, 8]:
      path_coverage[k] = float(np.mean(paths_found >= k))
  ```
- **Output**: Four floats per dataset.

#### M33: Average Path Length (of Successfully Mined Paths)

- **Definition**: Average number of **entity nodes** in the mined entity-relation paths. A path `[e0, r0, e1, r1, e2]` has 3 entity nodes (length 3 in nodes, 2 relation steps).
- **Rationale**: Longer paths capture more multi-hop context.
- **Computation**:
  ```python
  all_path_lengths = []
  for entity in range(num_entities):
      for path in entity_paths_dict.get(entity, []):
          # Path format: [e, r, e, r, ..., e] — entities at even indices
          num_entity_nodes = (len(path) + 1) // 2
          all_path_lengths.append(num_entity_nodes)
  
  avg_path_length = float(np.mean(all_path_lengths)) if all_path_lengths else 0.0
  median_path_length = float(np.median(all_path_lengths)) if all_path_lengths else 0.0
  ```
- **Output**: Two floats per dataset.

#### M34: Path Diversity (Distinct Relation Sequences)

- **Definition**: For each entity, extract the **relation sequence** (meta-path) from each of its mined entity-relation paths. Count the number of distinct relation sequences. Average across entities.
  
  path_diversity(e) = |{extract_relation_sequence(pᵢ) : pᵢ ∈ paths(e)}|
  
- **Reference**: Related to the concept of meta-paths in heterogeneous information networks [SunYuHan2011].
- **Rationale**: Two entity-relation paths may traverse different entities but follow the same relation pattern, providing redundant signal to the path encoder. Higher diversity = more varied evidence.
- **Computation**:
  ```python
  diversities = []
  for entity in range(num_entities):
      paths = entity_paths_dict.get(entity, [])
      if not paths:
          continue
      rel_sequences = set()
      for path in paths:
          rel_seq = extract_relation_sequence(path)
          rel_sequences.add(rel_seq)
      diversities.append(len(rel_sequences))
  
  avg_path_diversity = float(np.mean(diversities)) if diversities else 0.0
  ```
- **Output**: Single float per dataset.

#### M35: Average Number of Distinct Relation Types in Paths per Entity

- **Definition**: For each entity, across all its mined entity-relation paths, count how many distinct relation types appear. Average across entities.
  
  rel_types_in_paths(e) = |⋃ₚ {r : r appears in p}| for paths p of entity e
  
- **Rationale**: How much of the relational vocabulary is "reachable" via paths from a given entity.
- **Computation**:
  ```python
  rel_in_paths = []
  for entity in range(num_entities):
      paths = entity_paths_dict.get(entity, [])
      if not paths:
          continue
      rel_types = set()
      for path in paths:
          rel_seq = extract_relation_sequence(path)
          rel_types.update(rel_seq)
      rel_in_paths.append(len(rel_types))
  
  avg_rel_in_paths = float(np.mean(rel_in_paths)) if rel_in_paths else 0.0
  ```
- **Output**: Single float per dataset.

#### M36: Path Reachability (k-hop Reachability)

- **Definition**: Fraction of ordered entity pairs (e₁, e₂) reachable within k hops in the directed graph, for k ∈ {2, 3, 5, 10}. Estimated by sampling 10,000 random pairs.
- **Reference**: Standard reachability analysis [Newman2003].
- **Library**: Uses `nx.shortest_path_length` for individual pair queries.
- **Rationale**: If entities are unreachable even within 10 hops, entity-relation paths cannot carry information between them.
- **Computation**:
  ```python
  def estimate_reachability(G, num_samples=10000, k_values=[2, 3, 5, 10]):
      entities = list(G.nodes())
      results = {k: 0 for k in k_values}
      for _ in range(num_samples):
          src, tgt = random.sample(entities, 2)
          try:
              sp = nx.shortest_path_length(G, src, tgt)
              for k in k_values:
                  if sp <= k:
                      results[k] += 1
          except nx.NetworkXNoPath:
              pass
      return {k: count / num_samples for k, count in results.items()}
  ```
- **Output**: Four floats per dataset.

---

### CATEGORY 6: Connectivity and Global Structure

Macroscopic properties affecting path generation and information flow. All definitions follow standard graph theory [Newman2003].

#### M37: Number of Connected Components (Undirected)

- **Definition**: Number of connected components when treating all edges as undirected [Newman2003].
- **Library**: `nx.number_connected_components(G_undirected)`
- **Computation**:
  ```python
  num_components = nx.number_connected_components(G_undirected)
  ```
- **Output**: Single integer per dataset.

#### M38: Size of the Largest Connected Component (Undirected)

- **Definition**: |LCC| and |LCC| / |E| [Newman2003].
- **Library**: `nx.connected_components(G_undirected)`
- **Computation**:
  ```python
  largest_cc = max(nx.connected_components(G_undirected), key=len)
  lcc_size = len(largest_cc)
  lcc_fraction = lcc_size / num_entities
  ```
- **Output**: Integer and float per dataset.

#### M39: Number of Strongly Connected Components (Directed)

- **Definition**: Number of SCCs in the directed multigraph [Newman2003].
- **Library**: `nx.number_strongly_connected_components(G)`
- **Computation**:
  ```python
  num_scc = nx.number_strongly_connected_components(G)
  ```
- **Output**: Single integer per dataset.

#### M40: Size of the Largest Strongly Connected Component

- **Definition**: |LSCC| and |LSCC| / |E| [Newman2003].
- **Library**: `nx.strongly_connected_components(G)`
- **Computation**:
  ```python
  largest_scc = max(nx.strongly_connected_components(G), key=len)
  lscc_size = len(largest_scc)
  lscc_fraction = lscc_size / num_entities
  ```
- **Output**: Integer and float per dataset.

#### M41: Diameter (Approximate)

- **Definition**: The diameter (longest shortest path) of the LCC in the undirected graph [Newman2003]. If exact computation is too slow, approximate by sampling BFS from k random nodes.
- **Library**: `nx.diameter()` for small graphs; `nx.single_source_shortest_path_length()` for sampling.
- **Computation**:
  ```python
  lcc_subgraph = G_undirected.subgraph(largest_cc)
  
  # Try exact computation first; fall back to approximation for large graphs
  if lcc_size < 50000:
      try:
          diameter = nx.diameter(lcc_subgraph)
      except:
          diameter = _approximate_diameter(lcc_subgraph)
  else:
      diameter = _approximate_diameter(lcc_subgraph)
  
  def _approximate_diameter(G_sub, num_samples=200):
      """Lower bound on diameter via sampled BFS."""
      nodes = list(G_sub.nodes())
      max_dist = 0
      for _ in range(num_samples):
          source = random.choice(nodes)
          lengths = dict(nx.single_source_shortest_path_length(G_sub, source))
          max_dist = max(max_dist, max(lengths.values()))
      return max_dist
  ```
- **Output**: Single integer per dataset (exact or lower bound; annotate which).

#### M42: Average Shortest Path Length (Sampled)

- **Definition**: Average shortest path length between sampled pairs of entities in the LCC [Newman2003].
- **Library**: `nx.shortest_path_length` for individual queries.
- **Computation**:
  ```python
  def avg_shortest_path_sampled(G_sub, num_samples=10000):
      nodes = list(G_sub.nodes())
      distances = []
      for _ in range(num_samples):
          src, tgt = random.sample(nodes, 2)
          try:
              d = nx.shortest_path_length(G_sub, src, tgt)
              distances.append(d)
          except nx.NetworkXNoPath:
              pass
      return float(np.mean(distances)), float(np.std(distances))
  
  avg_sp, std_sp = avg_shortest_path_sampled(G_undirected.subgraph(largest_cc))
  ```
- **Output**: Two floats per dataset (mean and std).

#### M43: Average Clustering Coefficient

- **Definition**: The average local clustering coefficient [WattsStrogatz1998]:
  
  C̄ = (1/|V|) Σᵥ C(v)
  
  where C(v) = 2T(v) / (deg(v)(deg(v)−1)) and T(v) is the number of triangles through v. Nodes with degree < 2 have C(v) = 0 by convention.
  
- **Reference**: [WattsStrogatz1998], [Newman2003].
- **Library**: **MUST use** `nx.average_clustering(G_undirected)`.
- **Computation**:
  ```python
  avg_clustering = nx.average_clustering(G_undirected)
  ```
- **Output**: Single float per dataset in [0, 1].

#### M44: Global Clustering Coefficient (Transitivity)

- **Definition**: The graph transitivity [Newman2003]:
  
  T = 3 × |triangles| / |connected triples of nodes|
  
  This is distinct from the average local clustering coefficient (M43).
  
- **Reference**: [Newman2003].
- **Library**: **MUST use** `nx.transitivity(G_undirected)`.
- **Computation**:
  ```python
  transitivity = nx.transitivity(G_undirected)
  ```
- **Output**: Single float per dataset in [0, 1].

#### M45: Reciprocity

- **Definition**: 
  
  (a) **Untyped reciprocity**: fraction of triples (h, r, t) for which ∃r' such that (t, r', h) ∈ T.
  (b) **Typed reciprocity**: fraction of triples (h, r, t) for which (t, r, h) ∈ T.
  
  The NetworkX function `nx.overall_reciprocity(G)` computes the fraction of directed edges that have a reverse edge (ignoring edge types/keys in a multigraph). Use this as a cross-check for untyped reciprocity on the **simple directed graph**.
  
- **Reference**: [Garlaschelli2004] for the formal definition of reciprocity in directed networks.
- **Library**: Use `nx.overall_reciprocity()` on a simple DiGraph as a cross-check.
- **Computation**:
  ```python
  # Build lookup sets
  edge_pairs = set()
  typed_edges = set()
  for h, r, t in triples:
      edge_pairs.add((int(h), int(t)))
      typed_edges.add((int(h), int(r), int(t)))
  
  # Untyped reciprocity (over triples)
  reciprocal_count = sum(1 for h, r, t in triples if (int(t), int(h)) in edge_pairs)
  untyped_reciprocity = reciprocal_count / len(triples)
  
  # Typed reciprocity (same relation in reverse)
  typed_reciprocal = sum(1 for h, r, t in triples if (int(t), int(r), int(h)) in typed_edges)
  typed_reciprocity = typed_reciprocal / len(triples)
  
  # Cross-check with NetworkX on simple directed graph
  G_simple_directed = nx.DiGraph()
  for h, r, t in triples:
      G_simple_directed.add_edge(int(h), int(t))
  nx_reciprocity = nx.overall_reciprocity(G_simple_directed)
  # Note: nx_reciprocity is computed over unique directed edges, not triples,
  # so it may differ from untyped_reciprocity. Log both for comparison.
  ```
- **Output**: Two floats per dataset (untyped and typed reciprocity), plus nx_reciprocity for cross-check.

#### M46: Degree Assortativity

- **Definition**: Newman's degree assortativity coefficient [Newman2002]: the Pearson correlation of degrees at either end of an edge.
  
  r = [Σⱼₖ jk(eⱼₖ − qⱼqₖ)] / σ²_q
  
  where eⱼₖ is the fraction of edges connecting nodes of degree j and k, qⱼ is the excess degree distribution, and σ²_q is its variance.
  
- **Reference**: [Newman2002]. This is the standard measure of degree-degree correlation in networks.
- **Library**: **MUST use** `nx.degree_assortativity_coefficient(G_undirected)`.
- **Computation**:
  ```python
  assortativity = nx.degree_assortativity_coefficient(G_undirected)
  ```
- **Output**: Single float per dataset in [-1, 1].

---

### CATEGORY 7: Relation-Level Graph Properties

Properties of the relation-level structure, capturing how relations combine and interact.

#### M47: Relation-Specific Average Degree

- **Definition**: For each relation r, compute the average total degree of entities participating in triples with that relation:
  
  avg_degree_r = (1/|E_r|) Σ_{e ∈ E_r} deg(e)
  
  where E_r = {e : ∃t.(e,r,t) ∈ T or ∃h.(h,r,e) ∈ T}.
  
  Report the mean and std of avg_degree_r across all relations.
  
- **Rationale**: Relations connecting high-degree entities provide more context than relations connecting isolated entities.
- **Computation**:
  ```python
  relation_avg_degrees = []
  for r in range(num_relations):
      entities_in_r = set()
      for h, rel, t in triples:
          if int(rel) == r:
              entities_in_r.add(int(h))
              entities_in_r.add(int(t))
      if entities_in_r:
          degrees_r = [total_degrees[e] for e in entities_in_r]
          relation_avg_degrees.append(float(np.mean(degrees_r)))
      else:
          relation_avg_degrees.append(0.0)
  
  mean_rel_avg_degree = float(np.mean(relation_avg_degrees))
  std_rel_avg_degree = float(np.std(relation_avg_degrees))
  ```
- **Output**: Two floats per dataset, plus per-relation data.

#### M48: Relation Co-occurrence Matrix

- **Definition**: C[r₁, r₂] = |E_r₁ ∩ E_r₂| where E_r = {e : e participates in at least one triple with relation r}.
- **Rationale**: If relations always co-occur, they provide redundant signal. Diverse co-occurrence = more effective vocabulary.
- **Computation**:
  ```python
  relation_entities = defaultdict(set)
  for h, r, t in triples:
      relation_entities[int(r)].add(int(h))
      relation_entities[int(r)].add(int(t))
  
  cooccurrence = np.zeros((num_relations, num_relations), dtype=np.int32)
  for r1 in range(num_relations):
      for r2 in range(r1, num_relations):
          overlap = len(relation_entities[r1] & relation_entities[r2])
          cooccurrence[r1, r2] = overlap
          cooccurrence[r2, r1] = overlap
  ```
- **Output**: Matrix |R| × |R|, saved as numpy file. Report summary: average off-diagonal and fraction non-zero.

#### M49: Relation Co-occurrence Entropy

- **Definition**: Shannon entropy [Shannon1948] of the normalised upper-triangle of the co-occurrence matrix (excluding diagonal).
- **Library**: **MUST use** `scipy.stats.entropy`.
- **Computation**:
  ```python
  upper_tri = cooccurrence[np.triu_indices(num_relations, k=1)]
  upper_tri_nonzero = upper_tri[upper_tri > 0]
  
  if len(upper_tri_nonzero) > 0:
      probs = upper_tri_nonzero / upper_tri_nonzero.sum()
      cooccurrence_entropy = float(scipy_entropy(probs, base=2))
  else:
      cooccurrence_entropy = 0.0
  ```
- **Output**: Single float per dataset.

#### M50: Relation Graph Density

- **Definition**: Density of the relation co-occurrence graph: fraction of off-diagonal pairs with C[r₁,r₂] > 0.
  
  ρ = |{(r₁,r₂) : C[r₁,r₂] > 0, r₁ ≠ r₂}| / (|R|(|R|−1))
  
  Note: This counts ordered pairs. Equivalently, use unordered pairs with denominator |R|(|R|−1)/2.
  
- **Computation**:
  ```python
  num_cooccurring_pairs = int(np.sum(cooccurrence[np.triu_indices(num_relations, k=1)] > 0))
  max_pairs = num_relations * (num_relations - 1) / 2
  rel_graph_density = num_cooccurring_pairs / max_pairs if max_pairs > 0 else 0.0
  ```
- **Output**: Single float per dataset in [0, 1].

#### M51: Relation-Specific Reciprocity

- **Definition**: For each relation r:
  
  recip(r) = |{(h,r,t) ∈ T : ∃r'.(t,r',h) ∈ T}| / |{(h,r,t) ∈ T}|
  typed_recip(r) = |{(h,r,t) ∈ T : (t,r,h) ∈ T}| / |{(h,r,t) ∈ T}|
  
- **Reference**: Per-type extension of [Garlaschelli2004].
- **Computation**:
  ```python
  relation_recip = {}
  relation_typed_recip = {}
  
  for r in range(num_relations):
      r_triples = [(int(h), int(t)) for h, rel, t in triples if int(rel) == r]
      
      if not r_triples:
          relation_recip[r] = 0.0
          relation_typed_recip[r] = 0.0
          continue
      
      recip_count = sum(1 for h, t in r_triples if (t, h) in edge_pairs)
      typed_count = sum(1 for h, t in r_triples if (t, r, h) in typed_edges)
      
      relation_recip[r] = recip_count / len(r_triples)
      relation_typed_recip[r] = typed_count / len(r_triples)
  
  mean_rel_recip = float(np.mean(list(relation_recip.values())))
  std_rel_recip = float(np.std(list(relation_recip.values())))
  ```
- **Output**: Per-relation CSV plus aggregate stats.

---

### CATEGORY 8: Inductive Split-Specific Properties

These metrics characterise the Teru inductive splits [Teru2020]. Compute for both training and inference graphs of each version (V1–V4).

#### M52–M56

These metrics (Entity Overlap, Relation Overlap, Unique Relations in Inference Graph, Average Degree Comparison, Context Uniqueness in Inference Graph) are defined identically to the previous version. Compute using the same library functions specified for M7, M25, etc., but applied to the inductive split graphs.

**Key reference**: [Teru2020] for the split construction methodology.

---

### CATEGORY 9: Information-Theoretic Properties

These provide a principled characterisation of the graph's information content. All entropy computations follow [Shannon1948] and [Cover2006].

#### M57: Degree Distribution Entropy

- **Definition**: H(deg) = −Σₖ p(k) log₂ p(k) where p(k) = fraction of entities with total degree k.
- **Reference**: [Shannon1948].
- **Library**: **MUST use** `scipy.stats.entropy`.
- **Computation**:
  ```python
  degree_counts = np.bincount(total_degrees.astype(int))
  degree_probs = degree_counts[degree_counts > 0] / num_entities
  degree_entropy = float(scipy_entropy(degree_probs, base=2))
  ```
- **Output**: Single float per dataset.

#### M58: Conditional Entropy of Tail Given Relation H(t|r)

- **Definition**: Conditional entropy [Cover2006]:
  
  H(T|R) = Σᵣ p(r) H(T|R=r) = Σᵣ p(r) [−Σₜ p(t|r) log₂ p(t|r)]
  
  where p(r) = count(r)/|T|, and p(t|r) = count(r,t)/count(r) over triples (h,r,t).
  
- **Reference**: [Cover2006], Chapter 2.
- **Library**: **MUST use** `scipy.stats.entropy` for each conditional distribution H(T|R=r).
- **Rationale**: Lower H(T|R) means relations are more predictive of tails, which helps the framework.
- **Computation**:
  ```python
  from collections import Counter, defaultdict
  
  relation_tail_counts = defaultdict(Counter)
  for h, r, t in triples:
      relation_tail_counts[int(r)][int(t)] += 1
  
  H_t_given_r = 0.0
  total = len(triples)
  for r in range(num_relations):
      tail_counts = relation_tail_counts[r]
      r_total = sum(tail_counts.values())
      if r_total == 0:
          continue
      p_r = r_total / total
      tail_probs = np.array(list(tail_counts.values())) / r_total
      H_r = scipy_entropy(tail_probs, base=2)
      H_t_given_r += p_r * H_r
  ```
- **Output**: Single float per dataset.

#### M59: Conditional Entropy of Head Given Relation H(h|r)

- **Definition**: Same as M58 but for head entity distribution: H(H|R) = Σᵣ p(r) H(H|R=r).
- **Reference**: [Cover2006].
- **Library**: **MUST use** `scipy.stats.entropy`.
- **Computation**: Analogous to M58, collecting head counts instead.
- **Output**: Single float per dataset.

#### M60: Conditional Entropy of Relation Given (Head, Tail) H(r|h,t)

- **Definition**: H(R|H,T) = Σ_{h,t} p(h,t) H(R|H=h,T=t) [Cover2006].
- **Reference**: [Cover2006].
- **Library**: **MUST use** `scipy.stats.entropy`.
- **Rationale**: Directly measures relation prediction difficulty. Lower entropy = easier relation prediction.
- **Computation**:
  ```python
  pair_relation_counts = defaultdict(Counter)
  for h, r, t in triples:
      pair_relation_counts[(int(h), int(t))][int(r)] += 1
  
  H_r_given_ht = 0.0
  total = len(triples)
  for (h, t), rel_counts in pair_relation_counts.items():
      pair_total = sum(rel_counts.values())
      p_ht = pair_total / total
      rel_probs = np.array(list(rel_counts.values())) / pair_total
      H_ht = scipy_entropy(rel_probs, base=2)
      H_r_given_ht += p_ht * H_ht
  ```
- **Output**: Single float per dataset.

---

### CATEGORY 10: Multi-Relational Graph-Specific Properties

Properties specific to multi-relational (typed edge) graphs.

#### M61: Relation Multiplicity

- **Definition**: Average number of distinct relation types per connected directed entity pair (h,t).
  
  multiplicity(h,t) = |{r : (h,r,t) ∈ T}|
  avg_multiplicity = mean over all (h,t) with at least one triple.
  
- **Computation**:
  ```python
  pair_relations = defaultdict(set)
  for h, r, t in triples:
      pair_relations[(int(h), int(t))].add(int(r))
  
  multiplicities = [len(rels) for rels in pair_relations.values()]
  avg_multiplicity = float(np.mean(multiplicities))
  max_multiplicity = int(np.max(multiplicities))
  fraction_multi = float(np.mean(np.array(multiplicities) > 1))
  ```
- **Output**: Three values per dataset.

#### M62: Meta-Path Diversity (Length 2 and 3)

- **Definition**: A meta-path [SunYuHan2011] is a sequence of relation types. Count distinct meta-paths of length 2 and 3 that actually occur in the graph.
  
  - Length-2: (r₁, r₂) such that ∃h,m,t: (h,r₁,m) ∈ T and (m,r₂,t) ∈ T.
  - Length-3: sampled (100,000 random length-3 walks) to avoid combinatorial explosion.
  
  Report count and ratio to theoretical maximum (|R|² and |R|³).
  
- **Reference**: [SunYuHan2011] for the meta-path concept in heterogeneous networks.
- **Computation**:
  ```python
  # Length-2 meta-paths (exact)
  incoming_rels = defaultdict(set)
  outgoing_rels = defaultdict(set)
  for h, r, t in triples:
      outgoing_rels[int(h)].add(int(r))
      incoming_rels[int(t)].add(int(r))
  
  metapaths_2 = set()
  for m in range(num_entities):
      for r1 in incoming_rels.get(m, set()):
          for r2 in outgoing_rels.get(m, set()):
              metapaths_2.add((r1, r2))
  
  num_metapaths_2 = len(metapaths_2)
  metapath_2_ratio = num_metapaths_2 / (num_relations ** 2)
  
  # Length-3 meta-paths (sampled)
  metapaths_3 = set()
  for _ in range(100000):
      idx = random.randint(0, len(triples) - 1)
      h1, r1, t1 = int(triples[idx][0]), int(triples[idx][1]), int(triples[idx][2])
      out_edges = list(G.out_edges(t1, data=True))
      if not out_edges:
          continue
      _, t2, data2 = random.choice(out_edges)
      r2 = data2['relation']
      out_edges2 = list(G.out_edges(t2, data=True))
      if not out_edges2:
          continue
      _, t3, data3 = random.choice(out_edges2)
      r3 = data3['relation']
      metapaths_3.add((r1, r2, r3))
  
  num_metapaths_3 = len(metapaths_3)
  metapath_3_ratio = num_metapaths_3 / (num_relations ** 3)  # lower bound
  ```
- **Output**: Four values per dataset.

#### M63: Relation Symmetry Score

- **Definition**: For each relation r:
  
  symmetry(r) = |{(h,r,t) : (t,r,h) ∈ T}| / |{(h,r,t) ∈ T}|
  
  Report unweighted average and frequency-weighted average: Σᵣ p(r) × symmetry(r).
  
- **Reference**: Related to typed reciprocity [Garlaschelli2004].
- **Computation**: Reuses `relation_typed_recip` from M51.
  ```python
  symmetry_scores = list(relation_typed_recip.values())
  avg_symmetry = float(np.mean(symmetry_scores))
  weighted_symmetry = float(sum(
      (relation_counts[r] / len(triples)) * relation_typed_recip[r]
      for r in range(num_relations)
  ))
  ```
- **Output**: Two floats per dataset, plus per-relation data.

#### M64: Relation Composition Patterns

- **Definition**: Same as M62 length-2 count. Reported here for completeness.
- **Reference**: [SunYuHan2011].
- **Output**: Same as M62 length-2 count.

---

## Output Specification

### File Structure

```
output/
├── summary_metrics.csv                # One row per dataset, all scalar metrics
├── summary_metrics.json               # Same data in JSON format
├── degree_distributions/
│   ├── FB15k-237_degrees.csv          # entity_id, in_degree, out_degree, total_degree
│   ├── WN18RR_degrees.csv
│   ├── CoDEx-Large_degrees.csv
│   └── YAGO3-10_degrees.csv
├── relation_distributions/
│   ├── FB15k-237_relation_freq.csv    # relation_id, count, fraction
│   └── ...
├── context_analysis/
│   ├── FB15k-237_context_classes.csv  # context_hash, num_entities, example_entity_ids
│   └── ...
├── path_analysis/
│   ├── FB15k-237_paths_per_entity.csv # entity_id, num_paths_found, avg_path_length
│   └── ...
├── relation_cooccurrence/
│   ├── FB15k-237_cooccurrence.npy     # |R| x |R| numpy array
│   └── ...
├── inductive_splits/
│   ├── FB15k-237_V1_metrics.csv
│   └── ...
└── plots/
    ├── degree_distributions.png
    ├── relation_frequency.png
    ├── context_class_sizes.png
    └── ...
```

### Performance Data for Correlation

Include these performance metrics from the original paper for later correlation analysis (Section 7 of the paper):

```python
performance_data = {
    "FB15k-237": {
        "LP_MRR": 0.216, "LP_H10": 0.350, "LP_H1": 0.146,
        "RP_MRR": 0.972, "RP_H10": 0.998,
        "Effi_LP": 1.03, "params_M_LP": 0.21,
        "Effi_RP": 1.13, "params_M_RP": 0.86,
    },
    "WN18RR": {
        "LP_MRR": 0.069, "LP_H10": 0.124, "LP_H1": None,
        "RP_MRR": 0.874, "RP_H10": 0.999,
        "Effi_LP": 0.10, "params_M_LP": 0.67,
        "Effi_RP": 17.48, "params_M_RP": 0.05,
    },
    "CoDEx-Large": {
        "LP_MRR": 0.144, "LP_H10": 0.240, "LP_H1": 0.100,
        "RP_MRR": None, "RP_H10": None,
        "Effi_LP": 0.21, "params_M_LP": 0.68,
    },
    "YAGO3-10": {
        "LP_MRR": 0.060, "LP_H10": 0.093, "LP_H1": None,
        "RP_MRR": None, "RP_H10": None,
        "Effi_LP": 0.25, "params_M_LP": 0.24,
    },
}
```

---

## Computation Notes and Caveats

1. **Memory**: YAGO3-10 and CoDEx-Large are large. Use sparse representations where possible.

2. **Randomness**: Set `random.seed(42)` and `np.random.seed(42)` at the start for reproducibility.

3. **Runtime**: Path mining (M31–M35) for all entities on large datasets may take hours. Use multiprocessing.

4. **Edge cases**: Entities with degree 0 in training should be included in distributions but flagged.

5. **NetworkX limitations**: For large graphs, some functions may be slow. Use approximations where noted, documenting sample size.

6. **Inductive splits**: Each version (V1–V4) has distinct train/inference graphs. Compute M52–M56 for each.

7. **Library validation**: For any metric computed without a standard library function (M19 Gini, M24–M30 context analysis, M45 reciprocity, M61 multiplicity), include a **unit test on a small synthetic graph** with known correct values to validate the implementation. For example:
   - Gini: a uniform distribution [1,1,1,1] → G=0; a maximally unequal [0,0,0,4] → G=0.75 (for n=4).
   - Reciprocity: a symmetric graph should have typed_reciprocity = 1.0.
   - Context uniqueness: in a graph where every entity has a distinct neighbourhood, unique_fraction = 1.0.

8. **Cross-checks**: Where multiple formulations exist (e.g., two Gini formulas), compute both and assert they agree within floating-point tolerance (1e-10).

9. **Plots**: For each distribution-valued metric, generate a plot comparing all four datasets. Use consistent colour coding.

10. **Entity-relation path verification**: After mining, run `verify_path()` on all paths to ensure correct alternating structure. This is critical because subsequent metrics (M34, M35) depend on correctly extracting relation vs. entity elements from the path.
