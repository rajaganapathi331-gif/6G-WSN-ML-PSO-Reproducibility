# Methods and Executable Mathematical Formulation

This document defines the quantities implemented in code so that the revised Methods section can use consistent notation.

## 1. First-order radio model

For a packet of \(L\) bits transmitted over distance \(d\),

\[
E_{tx}(L,d)=
\begin{cases}
L E_{elec} + L \epsilon_{fs}d^2, & d < d_0 \\
L E_{elec} + L \epsilon_{mp}d^4, & d \ge d_0
\end{cases}
\]

where

\[
d_0 = \sqrt{\epsilon_{fs}/\epsilon_{mp}}.
\]

Reception and aggregation are

\[
E_{rx}(L)=L E_{elec},
\qquad
E_{da}(L)=L E_{DA}.
\]

## 2. Markov node-state prediction

Let the state space be

\[
\mathcal{S}=\{\text{sleep},\text{idle},\text{rx},\text{tx}\}.
\]

The empirical transition probability is

\[
P_{ij} =
\frac{N_{ij}+\alpha}
{\sum_k (N_{ik}+\alpha)},
\]

where \(N_{ij}\) is the observed count from state \(i\) to \(j\) and \(\alpha\) is Laplace smoothing.

For current state \(s_t=i\), expected one-step energy drain is

\[
\hat{e}^{M}_{t+1}
=
\sum_j P_{ij} c_j,
\]

where \(c_j\) is the nominal state-energy cost.

## 3. Supervised ML energy predictor

For node \(i\) at time \(t\), the feature vector is

\[
x_{i,t} =
[
E^{res}_{i,t},
d^{sink}_{i},
deg_i,
q_{i,t},
\tau_{i,t},
T_t,
\ell_{i,t},
\mu^{drain}_{i,t},
\sigma^{drain}_{i,t},
\hat{e}^{M}_{i,t+1}
].
\]

The target is next-step measured/simulated drain

\[
y_{i,t}=E^{drain}_{i,t+1}.
\]

The repository uses a Random Forest regressor with group-held-out node validation. The trained predictor produces

\[
\hat{e}^{ML}_{i,t+1}=f_\theta(x_{i,t}).
\]

PSO remains an optimizer; it is not described as the ML model.

## 4. Energy-aware cluster-head score

Within each spatial cluster, candidate node \(i\) receives score

\[
S_i =
w_E \tilde{E}^{res}_i
+
w_L \tilde{\ell}_i
+
w_C \tilde{C}_i
-
w_D \tilde{\hat e}_i
-
w_Q \tilde{q}_i ,
\]

where normalized terms represent residual energy, link quality, centrality, predicted drain, and traffic load. The maximum-scoring alive node becomes the cluster head.

## 5. PSO routing-policy optimization

A particle contains a bounded vector of raw routing weights \(z\). The router converts it to a simplex vector using

\[
w_j = \frac{\exp(z_j)}{\sum_k \exp(z_k)}.
\]

For candidate next hop \(v\) from node \(u\), the routing feature vector is

\[
\phi(u,v) =
[
\tilde d_{uv},
1-\tilde E^{res}_v,
\tilde{\hat e}_v,
1-\tilde r_{uv},
\tilde \tau_{uv},
\tilde p_{uv}
],
\]

where the last term penalizes insufficient progress toward the sink.

The local edge score is

\[
C(u,v;w)=w^T \phi(u,v).
\]

The PSO objective is the average selected edge cost over a reproducible subset of alive source nodes plus penalties for route infeasibility and QoS violations:

\[
J(w)=
\frac{1}{|\Omega|}
\sum_{u\in\Omega} \min_{v\in\mathcal{N}(u)} C(u,v;w)
+
\lambda_f P_{fail}
+
\lambda_q P_{QoS}.
\]

PSO updates particle velocity and position as

\[
v_k^{t+1}
=
\omega v_k^t
+
c_1 r_1(p_k-x_k^t)
+
c_2 r_2(g-x_k^t),
\]

\[
x_k^{t+1}=x_k^t+v_k^{t+1}.
\]

## 6. Primary network metrics

Packet-delivery ratio:

\[
PDR = \frac{N_{delivered}}{N_{generated}}.
\]

Energy fairness (Jain index):

\[
J_E =
\frac{(\sum_i E_i)^2}
{n\sum_i E_i^2}.
\]

Delivered bits per joule:

\[
\eta_E =
\frac{L N_{delivered}}
{E_{initial,total}-E_{remaining,total}}.
\]

Network-lifetime landmarks are reported as first-node-death (FND), half-node-death (HND), and final alive-node count.

## 7. Statistical analysis

For each protocol and metric, confidence intervals are obtained by bootstrap resampling across independent seeds. The proposed-vs-baseline comparison uses a paired Wilcoxon signed-rank test on identical topology seeds. Multiplicity is controlled by the Benjamini–Hochberg procedure.

All variables in the implementation use the same meanings defined here.
