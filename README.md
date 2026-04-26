# Hypernetworks for Knowledge Distillation in Image Classification

A regular classification model is used to generate *a priori* class probabilities.  
These probabilities are then used as input to a hypernetwork, which generates specialized weights for a second classification model.

This enables **per-sample adaptation** of the network based on auxiliary predictive signals.

---

## 🧪 Model Variants

We compare several architectures to study the impact of conditional weight generation:

- **Regular ResNets**  
  Standard convolutional networks with fixed weights (baseline).

- **Pure Hypernetwork ResNets**  
  All weights are generated from an auxiliary input \( z \):  
  \[
  W = \Delta W(z)
  \]

- **Residual Hypernetwork ResNets**  
  Static base weights with additive hypernetwork updates:  
  \[
  W = W_{\text{base}} + \alpha \cdot \Delta W(z)
  \]

- **Gated Hypernetwork ResNets**  
  Learnable interpolation between static and dynamic weights:  
  \[
  W = (1 - g)\, W_{\text{base}} + g\, \Delta W(z)
  \]

- **Modulated ResNets (Feature-wise Modulation)**  
  Low-dimensional conditioning of activations:  
  \[
  y = \gamma(z)\, f(x) + \beta(z)
  \]

---

These variants allow us to analyze trade-offs between **stability**, **expressiveness**, and **conditional adaptation**.

---

## ⚠️ Limitations & Failure Modes

### Shortcut Learning in Hypernetworks

When conditioning on a strong auxiliary signal \( z \) (e.g., teacher predictions), the model may **ignore the image input entirely**.

Instead of learning:
\[
y = f(x, z)
\]

the model can collapse to:
\[
y \approx f(z)
\]

resulting in the shortcut:

z → weights → logits

### 🔬 Observations

- Accuracy can remain high even when `x = 0`  
- The model may rely heavily on \( z \), even with partial hypernetworks  
- This effect appears even when conditioning only later layers  

---

### 🧪 Diagnostic Check

| Test               | Expected Behavior     |
|--------------------|----------------------|
| `x = 0, z = real` | Accuracy should drop |
| `x = real, z = 0` | Accuracy should drop |
| Both inputs used  | Highest accuracy     |

---

### 🛠️ Mitigation Strategies (Ongoing Work)

We investigate several approaches to mitigate shortcut learning:

- Residual and gated formulations    
- Regularization (dropout, noise injection in \( z \))  
- Restricting hypernetworks to later layers  
- **Joint training of predictor and hypernetwork:**  
  Training both components simultaneously yields the most stable behavior. In this regime, the predictor introduces weaker signals early in training, encouraging reliance on image features.
- **Prediction softening (temperature scaling / smoothing):**  
  Softening the predictor outputs (e.g., via temperature-scaled softmax) reduces shortcut reliance and improves performance. However, this approach is sensitive to hyperparameters and requires careful tuning.

Previous approaches were effective for models using GroupNorm. However when switching to BatchNorm, those approaches weren't enough, given the per-sample nature of hyper layers. 

- **Custom Loss Function:**
  Give hypernet greater focus to samples in which the prior has worse performance
- **Custom Data augmentation:**
  Generates disagreement between prior and hypernet, making the later not fully trust predictor.
- **Training schedulinng:**
  Gives prior an early disadvantage, making it less attractive for bypassing.
- **Weight initialization:**
  Makes the model focus more on input than conditioning during early epochs.

---

## 🔎 Related Work

This project builds on ideas from:

- Hypernetworks (Ha et al., 2017)  
- Dynamic convolution (CondConv, Dynamic Filter Networks)  
- Knowledge distillation (Hinton et al., 2015)  

Unlike standard distillation, we use teacher predictions to **dynamically generate weights**, enabling **input-conditioned models**.

---

## 🚧 Current Status

- ✔ Core architectures implemented  
- ✔ Shortcut learning issue identified and reproduced  
- 🔬 Mitigation strategies under investigation  
- 🚧 Ongoing experimentation and refinement  

---

## 📌 Future Work

- Improved architectural constraints for stable conditioning  
- Better regularization of hypernetworks  
- Extended evaluation on larger datasets  
- Analysis of generalization vs. conditional capacity trade-offs
