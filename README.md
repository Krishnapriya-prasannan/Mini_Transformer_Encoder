
### **Day 15 - Build Your Own Transformer Model (Mini BERT)**  
As part of my **#100DaysOfAI** challenge, on **Day 15**, I implemented a **Mini Transformer Encoder** model inspired by **BERT**. This compact version helps understand the internals of the Transformer architecture by building it from scratch, including **positional encoding**, **multi-head self-attention**, and **encoder blocks** using **synthetic data**.

---

### **Goal**  
Understand and implement the **core components of a Transformer Encoder** architecture from scratch to build a simplified BERT-like model for sequence processing.

---

### **Technologies Used**

| Tool/Library | Purpose                                              |
|--------------|------------------------------------------------------|
| Python       | Core programming language                            |
| PyTorch      | Building and training deep learning models           |
| NumPy        | Efficient numerical operations                       |
| Torch.nn     | Neural network components like Linear, LayerNorm     |
| Torch.Tensor | Working with input sequences and synthetic data      |

---

### **How It Works**

1. **Positional Encoding**
   - Injects positional information into token embeddings using sine and cosine functions of different frequencies.

2. **Multi-Head Self-Attention**
   - Projects inputs into queries, keys, and values.
   - Computes scaled dot-product attention across multiple heads for richer representations.

3. **Encoder Block**
   - Applies multi-head self-attention followed by a feed-forward network.
   - Each sub-layer is followed by residual connections and layer normalization.

4. **Mini BERT Architecture**
   - Embeds input tokens and adds positional encodings.
   - Passes through stacked encoder layers (2 layers used here).
   - Produces contextualized embeddings as output.

5. **Synthetic Data Input**
   - Randomly generated token IDs simulate real text input.
   - Used to test model architecture and ensure correct tensor shapes.

---

### **Highlights**

- Built core **Transformer Encoder** components from scratch.
- Gained hands-on understanding of **multi-head attention** and **residual connections**.
- Implemented **layer normalization** and **position encoding** manually.
- Created a clean and modular architecture resembling a mini-BERT encoder.
- Verified working model with synthetic data and confirmed output dimensions.

---

### **What I Learned**

- Deep understanding of how **self-attention** mechanisms compute relevance between tokens.
- Why **positional encoding** is crucial in models without recurrence.
- How multiple attention heads help capture diverse features from sequences.
- Practical experience in building and stacking transformer blocks.
- Foundations to extend this into **classification**, **language modeling**, or **masked prediction** tasks.

---

