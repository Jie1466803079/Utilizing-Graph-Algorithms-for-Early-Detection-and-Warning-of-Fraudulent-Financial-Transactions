# Graph Algorithms for Early Detection and Warning of Fraudulent Financial Transactions

> Detects fraudulent mobile-money transactions by combining transaction-network (graph) analysis with clustering, graph-embedding, graph-neural-network and tree-based classifiers, structured around the Big Data Analytics Lifecycle.
>
> **Author led a six-member team on this project.**

`Python` · `NetworkX` · `StellarGraph (node2vec, GCN)` · `scikit-learn` · `XGBoost`

## Problem

Mobile money lets people make financial transactions by phone and is spreading quickly, especially in developing countries with weaker institutional infrastructure — but that convenience also increases the opportunity for fraud. Legitimate mobile-money datasets are scarce and private, so this project works with the PaySim synthetic transaction dataset (scaled down to 95 time steps, i.e. under four days of simulated activity). The goal is to detect fraudulent transactions and give early-warning recommendations to financial service providers.

## Approach

1. **Data preparation** — Load the PaySim dataset (1,048,575 transactions × 11 columns); confirm no duplicates and no missing values; inspect outliers; engineer features (transferer/transferee labels, customer→customer / customer→merchant flow type, hour-of-day, one-hot transaction type); drop `isFlaggedFraud`.
2. **Class-imbalance handling** — The data is highly skewed (1,142 fraud vs 1,047,433 valid). Valid transactions are down-sampled to build a balanced set of 16,736 transactions.
3. **Graph construction (NetworkX)** — Build a directed transaction network `G` (18,827 nodes, 16,574 edges) where accounts are nodes and transactions are edges, plus an amount-weighted network `G_T` and a transaction-count-weighted network `G_N`.
4. **Network analysis** — Compare fraud vs normal accounts on in/out-degree, in/out-strength, edge-weight distributions, average-neighbour degree/strength, clustering coefficient, community detection (greedy modularity), k-core, betweenness centrality and PageRank; trace connectivity with a custom snowball-sampling function; generate per-node features for classification.
5. **Modelling** — Seven models: K-means clustering; node2vec embeddings (biased random walks with `length=10, n=5, p=0.5, q=2.0`, then 128-dimensional Word2Vec vectors) classified with `LogisticRegressionCV`; a StellarGraph GCN; an MLP artificial neural network; Random Forest; and XGBoost.

## Results

Verified from notebook outputs:

- **Dataset & imbalance:** 1,048,575 transactions; 1,142 fraudulent vs 1,047,433 valid.
- **Fraud pattern:** all 1,142 fraudulent transactions are customer-to-customer; the notebook's type analysis and conclusion state fraud occurs only in `TRANSFER` and `CASH_OUT` transactions.
- **Balanced sample:** 16,736 transactions used for modelling.
- **Transaction graph:** `G` = 18,827 nodes / 16,574 edges (same node/edge counts for `G_T` and `G_N`).

Reported model performance (from the notebook's analysis):

- node2vec + `LogisticRegressionCV` embeddings: ~93% accuracy on the test set.
- ANN (MLP): ~98% accuracy on the balanced sample, versus ~86% valid-transaction accuracy and ~0.76 AUC on the full imbalanced data.
- Random Forest: AUC > 0.91 (accuracy close to 1).
- XGBoost: AUC 0.90.
- GCN: underperformed (near-zero loss, low accuracy) and was set aside.

**Early-warning recommendations** the notebook draws for financial providers: watch `TRANSFER`/`CASH_OUT` transactions; monitor accounts that transact with a large number of counterparties; scrutinise abnormally large single-transaction amounts; monitor accounts with very large total inflow/outflow; and pay attention to customer-to-customer transactions.

All figures are embedded inside the notebook; there are no standalone image files in the repository.

## Repository structure

```
.
├── Fraudulent Financial Transaction Detection.ipynb   # full pipeline: data prep, graph analysis, 7 models
└── README.md
```

## Tech

- **Language / environment:** Python (Google Colab / Jupyter notebook)
- **Data & visualisation:** pandas, NumPy, seaborn, matplotlib, SciPy
- **Graph:** NetworkX (degree/strength, greedy-modularity communities, k-core, betweenness centrality, PageRank), StellarGraph 1.2.1 (node2vec `BiasedRandomWalk`, `GCN`), gensim `Word2Vec`
- **Machine learning:** scikit-learn (K-means, `LogisticRegressionCV`, `MLPClassifier`, Random Forest, and others), XGBoost, TensorFlow/Keras (GCN training)

## Team & role

The repository states the author led a six-member team on this project.
