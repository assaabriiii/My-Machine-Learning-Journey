# AI + ML + Recommender Systems Roadmap (Projects & Papers)

This roadmap is designed for an AI Master’s student who already has **tabular ML projects**, **basic CNN classification projects**, **a small NLP notebook**, and some **from-scratch basics**. It avoids duplicating what you’ve already built (CIFAR10 / cats-vs-dogs / emotion / traffic signs / rock-paper-scissors CNN, Titanic/heart disease/housing/tips, SMOTE, basic NLTK sentiment, LinearRegression + simple NeuralNetwork).

---

## How to use this roadmap

### The loop (repeat for each topic)
1. **Read 1–3 anchor papers** (below) + a short blog/tutorial to fill gaps.
2. **Reproduce** one key result (even partially).
3. **Extend** it with *one* meaningful twist (ablation, new dataset, new loss, new evaluation).
4. **Package** the project: clean repo, README, config, reproducibility, and a short write-up.

### Deliverables checklist for every serious project
- [ ] Clean dataset loading + train/val/test split (and leakage checks)
- [ ] Baselines (at least 2)
- [ ] Correct metrics + confidence intervals or repeated runs when feasible
- [ ] Error analysis (where it fails, why)
- [ ] Reproducible run command + pinned dependencies
- [ ] Short write-up: what worked, what didn’t, what you’d try next

---

# Part A — Projects Roadmap (General AI / ML)

## A0. Upgrade your repo into a “research-ready” workspace (high leverage)
**Goal:** make every future project easier to run, compare, and reuse.

### Project A0.1 — Experiment template repo (1 time, reused forever)
**Build**
- A standard structure:
  - `src/` (model + training code)
  - `configs/` (yaml for experiments)
  - `notebooks/` (only for exploration)
  - `reports/` (paper notes + results)
  - `data/` (gitignored; use a download script)
- Add: `pyproject.toml` or `requirements.txt`, `pre-commit`, `ruff/black`, `pytest`
- Add a simple experiment tracker:
  - minimal: CSV logging
  - better: Weights & Biases or MLflow

**Why it matters**
- You’ll do *more experiments* with *less friction*.

---

## A1. “From scratch” foundations you probably haven’t done yet
You already have Linear Regression and a Neural Network notebook; the next step is to fill in the “missing classical core” and key mathy algorithms.

### Project A1.1 — Logistic Regression + Softmax Regression from scratch
**Build**
- Binary logistic regression with L2 regularization (GD + Newton/IRLS)
- Multiclass softmax regression + cross-entropy
- Calibration check (reliability curves)

**Key skills**
- Optimization, regularization, probabilistic interpretation, calibration

### Project A1.2 — Decision Tree + Random Forest (minimal but correct)
**Build**
- CART-style tree (Gini/entropy), pruning basics
- Random forest with bootstrap + feature subsampling
- Compare to scikit-learn

**Key skills**
- Greedy splitting, variance reduction, interpretability

### Project A1.3 — k-means + PCA + Gaussian Mixture Model (EM)
**Build**
- k-means (k-means++ init)
- PCA via SVD (and whitening)
- GMM with EM + model selection (AIC/BIC)

**Key skills**
- Unsupervised learning, latent variables, EM

### Project A1.4 — Mini autodiff engine + MLP trainer (micrograd-style)
**Build**
- Scalars or small tensors + backprop graph
- MLP + SGD/Adam + weight decay
- Validate gradients with finite differences

**Key skills**
- Backprop “for real” and debugging deep learning

---

## A2. Modern deep learning projects (not basic classification)
### Project A2.1 — Self-supervised representation learning (SimCLR-style)
**Build**
- Contrastive training on CIFAR-10/100 or STL-10
- Linear probe evaluation + fine-tuning comparison

**Key skills**
- Augmentations, contrastive losses, representation evaluation

### Project A2.2 — Segmentation (U-Net) or Detection (RetinaNet/YOLO-lite)
**Build**
- U-Net for medical or satellite segmentation (or Cityscapes subset)
- OR a simple object detector on a lightweight dataset
- Strong focus on metrics (IoU / mAP), not just “it trains”

**Key skills**
- Structured prediction, losses, evaluation

### Project A2.3 — Diffusion model (small) for images
**Build**
- DDPM on MNIST / Fashion-MNIST / small 32×32 dataset
- Sampling + guidance experiment (even a tiny one)

**Key skills**
- Noise schedules, denoising objective, generative evaluation

---

## A3. NLP beyond NLTK
### Project A3.1 — Transformer fine-tuning + dataset shift
**Build**
- Fine-tune a small transformer on IMDB
- Test on out-of-domain sentiment dataset
- Do error analysis + calibration

**Key skills**
- Tokenization, fine-tuning, robustness

### Project A3.2 — Retrieval Augmented Generation (RAG) mini-system
**Build**
- Index a small corpus (your papers folder, lecture notes, docs)
- Retrieval (BM25 + embeddings) + generation + citations
- Evaluate with a small “gold” QA set

**Key skills**
- Search, embeddings, evaluation, prompt+system design

---

## A4. Reinforcement learning (super valuable for recommender systems later)
### Project A4.1 — Bandits (multi-armed + contextual)
**Build**
- ε-greedy, UCB, Thompson Sampling
- Contextual bandits: LinUCB
- Offline simulation + regret plots

### Project A4.2 — Deep RL baseline
**Build**
- DQN on CartPole
- PPO on a simple Gym environment
- Focus on stability + reproducibility (seeds, averages)

---

# Part B — Recommender Systems Projects Roadmap (Specialization)

Use **MovieLens** (explicit ratings), then **implicit feedback** datasets (e.g., Amazon reviews / Steam / Last.fm), then a **sequence** dataset if possible.

## B0. Evaluation + data plumbing (do this first)
### Project B0.1 — Recsys evaluation harness (your “metrics engine”)
**Build**
- Train/val/test splits for:
  - explicit: rating prediction
  - implicit: leave-one-out or time-based
- Metrics:
  - Rating: RMSE/MAE (but don’t stop there)
  - Ranking: Recall@K, Precision@K, NDCG@K, MAP@K, MRR
  - Coverage, novelty, diversity (optional but impressive)
- Negative sampling strategies and their pitfalls

**Deliverable**
- A reusable library `src/recsys/metrics.py` + `src/recsys/splits.py`

---

## B1. Strong baselines (you must beat these)
### Project B1.1 — Popularity + item-item CF + user-user CF
**Build**
- Popularity baseline (most popular items)
- Item-item cosine similarity (implicit and explicit variants)
- User-user kNN baseline

**Why**
- Many fancy models don’t beat well-tuned baselines.

**Write-up focus**
- When does each baseline fail? (cold-start, sparsity, niche items)

---

## B2. Matrix factorization era (still incredibly relevant)
### Project B2.1 — Explicit MF (SGD) + biases
**Build**
- Biased MF: global + user bias + item bias + dot product
- Regularization sweep + early stopping
- Compare to Surprise library

### Project B2.2 — Implicit MF (ALS) + BPR
**Build**
- ALS for implicit feedback
- Bayesian Personalized Ranking (pairwise loss)
- Compare ALS vs BPR on ranking metrics

**Write-up focus**
- How negative sampling affects results
- Popularity bias and long-tail

---

## B3. Feature-aware models (bridging classical ↔ deep)
### Project B3.1 — Factorization Machines (FM) + Field-aware variants (optional)
**Build**
- FM with sparse features (user, item, genre, tags)
- Compare vs MF on cold-start-ish settings

### Project B3.2 — Wide & Deep / DeepFM (industrial classic)
**Build**
- Wide & Deep baseline + DeepFM
- Focus on feature engineering, embeddings, regularization

---

## B4. Two-stage recommenders (retrieval + ranking) — “real-world” pattern
### Project B4.1 — Candidate generation with two-tower retrieval
**Build**
- Two-tower model to embed users and items
- ANN retrieval with FAISS
- Evaluate recall@K retrieval

### Project B4.2 — Learning-to-rank re-ranker
**Build**
- Re-rank top N candidates with a ranking model (e.g., MLP, LightGBM ranker)
- Compare pointwise vs pairwise losses
- Evaluate NDCG@K improvements

**Write-up focus**
- Separation of concerns: retrieval vs ranking
- Latency/compute tradeoffs

---

## B5. Sequential / session-based recommendation
### Project B5.1 — GRU4Rec baseline
### Project B5.2 — SASRec or BERT4Rec (self-attention for sequences)
**Build**
- Next-item prediction
- Compare RNN vs attention on sequence length sensitivity

---

## B6. Graph-based recommendation
### Project B6.1 — LightGCN reproduction
**Build**
- Graph construction, sampling, training
- Compare vs MF/BPR on sparse regimes

---

## B7. “Decision-making” recommenders (bandits + RL)
### Project B7.1 — Contextual bandit ranking simulation
**Build**
- Position bias in clicks
- Learn a policy that improves CTR in simulation
- Offline evaluation caveats (IPS / DR)

---

# Part C — Paper Roadmap (General AI + Recommender Systems)

## C0. Fixing your current paper list (what to change)
Your current “20 Revolutionary AI Papers” list is a **solid start**, but it mixes:
- **actual papers** with **topics** (Part 2 is mostly themes, not citations),
- some **overconfident claims** (“ended CNN dominance”, etc.),
- and it’s missing a few “boring but critical” building blocks (BatchNorm, Adam, word2vec, DQN/PPO, diffusion, contrastive learning).

If you keep that list, I’d rewrite it into **(a) Foundations**, **(b) Training/Optimization**, **(c) Representation Learning**, **(d) Generative**, **(e) RL**, **(f) Recommenders**, and for each item include:
- 1–2 sentence summary
- what to reproduce
- what to extend

---

## C1. General AI paper sequence (high ROI)
### Foundations (architecture + training)
1. **Dropout** (Srivastava et al., 2014) → regularization mindset
2. **Batch Normalization** (Ioffe & Szegedy, 2015) → training stability
3. **Adam** (Kingma & Ba, 2015) → optimizer baseline
4. **ResNet** (He et al., 2015) → deep training + skip connections
5. **Attention / Seq2Seq with attention** (Bahdanau et al., 2015) → why attention matters
6. **Attention Is All You Need** (Vaswani et al., 2017) → transformers

### Representation learning
7. **word2vec** (Mikolov et al., 2013) → embeddings intuition
8. **SimCLR** (Chen et al., 2020) *or* **MoCo** (He et al., 2020) → contrastive learning
9. **BERT** (Devlin et al., 2018) → pretrain/fine-tune paradigm

### Generative modeling
10. **VAE** (Kingma & Welling, 2013) → latent variable modeling
11. **GAN** (Goodfellow et al., 2014) → adversarial training
12. **DDPM** (Ho et al., 2020) → diffusion basics

### RL basics
13. **DQN** (Mnih et al., 2015) → deep RL
14. **PPO** (Schulman et al., 2017) → stable policy gradients

**What to build after C1**
- Pick 1 paper per category and implement a minimal reproduction (see Part D template).

---

## C2. Recommender Systems paper sequence (specialization)
### Classical + evaluation (do not skip)
1. **Netflix Prize MF**: “Matrix Factorization Techniques for Recommender Systems” (Koren, Bell, Volinsky, 2009)
2. **Implicit feedback**: “Collaborative Filtering for Implicit Feedback Datasets” (Hu, Koren, Volinsky, 2008)
3. **BPR**: “Bayesian Personalized Ranking from Implicit Feedback” (Rendle et al., 2009)
4. **Factorization Machines** (Rendle, 2010)
5. **Evaluation beyond RMSE**: read at least one ranking-metrics/evaluation-protocols paper or survey

**Build with these**
- Your Projects B2 + B3 + the evaluation harness (B0)

### Deep learning + industry patterns
6. **Neural Collaborative Filtering** (He et al., 2017)
7. **Wide & Deep** (Cheng et al., 2016)
8. **DeepFM** (Guo et al., 2017)
9. **YouTube Recommendations** (Covington et al., 2016) — two-stage retrieval+ranking
10. **DIN** (Zhou et al., 2018) — attention over user history

**Build with these**
- Two-tower retrieval + re-ranker (B4), plus feature-aware models (B3)

### Sequential recommendation
11. **GRU4Rec** (Hidasi et al., 2015/2016)
12. **SASRec** (Kang & McAuley, 2018)
13. **BERT4Rec** (Sun et al., 2019)

**Build with these**
- Next-item prediction project (B5)

### Graph recommenders
14. **NGCF** (Wang et al., 2019) *or* **LightGCN** (He et al., 2020)

**Build with these**
- Graph-based recommender (B6)

### Bandits / counterfactual evaluation (advanced but differentiating)
15. Start with a survey or tutorial paper on **counterfactual evaluation** (IPS / DR) for recommenders.
16. A contextual bandit paper (LinUCB baseline) applied to ranking/click models.

**Build with these**
- Bandit simulation with position bias (B7)

---

# Part D — What to write + do after reading each paper (template)

## D1. The “Paper Card” (1 page, every time)
Create `papers/notes/<year>_<shortname>.md` with:

### 1) Citation + links
- Title, authors, venue, year
- PDF link
- Code link (if any)

### 2) 3-sentence summary
- Problem
- Core idea
- Key result

### 3) What’s *actually new*?
- New objective? new architecture? new training trick? new evaluation protocol?

### 4) Method in your own words
- Inputs → transformations → outputs
- Loss function (write it)
- Training recipe (optimizer, batch size, sampling)

### 5) Reproduction plan (minimum viable reproduction)
- Dataset
- Baseline(s)
- Metric(s)
- One figure/table you’ll recreate

### 6) Gotchas / assumptions
- Hidden dependencies, data leakage risks, evaluation protocol traps

### 7) Critique
- Strengths
- Weaknesses
- What you don’t buy yet

### 8) Extension idea (must be concrete)
Pick ONE:
- ablation (remove a component)
- new dataset/domain
- efficiency improvement
- fairness/bias analysis
- robustness test

### 9) “If I had 1 more week…”
List 3 next experiments.

---

## D2. The “Paper → Code” mapping (what you build)
For each paper you choose to implement, create:
- `projects/<topic>/<paper_shortname>/`
  - `README.md` (what you reproduced + what you extended)
  - `train.py` (config-driven)
  - `evaluate.py`
  - `results/` (saved metrics + plots)
  - `report.md` (your conclusions)

**Rule:** If you can’t reproduce the main trend, document *why* (and what you tried). That is still valuable.

---

# Part E — Suggested files to add to your repo

## E1. Paper notes folder structure
```
papers/
  roadmap.md
  notes/
    2009_koren_mf.md
    2008_hu_implicit_als.md
    2009_rendle_bpr.md
    2016_covington_youtube.md
    ...
```

## E2. Projects folder structure
```
projects/
  recsys/
    00_eval_harness/
    01_baselines_knn/
    02_mf_als_bpr/
    03_fm_deepfm/
    04_two_tower_retrieval/
    05_sequential_sasrec/
    06_lightgcn/
    07_bandits_sim/
  general/
    autodiff_micrograd/
    ssl_simclr/
    diffusion_ddpm/
    segmentation_unet/
    rag_mini/
    rl_dqn_ppo/
```

---

# Part F — “Pick-your-next-3” suggestions (based on what you’ve already done)

If you want the fastest path to **stronger overall AI skills + a recsys specialty**, do these next:

1. **B0.1 Evaluation harness** (reusable + signals maturity)
2. **B2.2 Implicit ALS + BPR** (core recsys competence)
3. **B4.1 Two-tower retrieval + FAISS** (industry pattern, very hireable)
4. (bonus) **A4.1 Bandits** (ties directly to recommendation decision-making)

