# Prototypical Networks

## With CNN-Transformer encoder and tokenisation

- DNA sequences are cleaned, non-ACGT characters removed, and mechanism labels encoded before splitting into train, validation, and test sets.  
- Sequences are tokenised into overlapping k-mers, converted to integer IDs, and padded to a fixed maximum length.  
- Episodic few-shot tasks are created using N-way K-shot Q-query sampling to form support and query sets per mechanism.  
- A Prototypical Network with CNN and Transformer layers is trained using cosine similarity between query embeddings and class prototypes.  
- Model performance is evaluated on validation and test tasks, and visualised through accuracy curves and t-SNE plots of learned embeddings.  

![model results](image.png)
![t-SNE projection](image-1.png)