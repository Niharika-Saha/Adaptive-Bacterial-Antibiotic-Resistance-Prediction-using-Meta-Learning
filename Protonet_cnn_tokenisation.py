# ProtoNet + CNN + Transformer for DNA mechanisms (tokenised k-mers)

import os, random, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

from google.colab import files
uploaded = files.upload()
df = pd.read_csv(list(uploaded.keys())[0])
print("Raw dataset loaded:", df.shape)
df = df.dropna(subset=["sequence","mechanism"]).reset_index(drop=True)
df["sequence"] = df["sequence"].str.upper().str.replace(r"[^ACGT]", "", regex=True)
print("Cleaned dataset:", df.shape, " | unique mechanisms:", df["mechanism"].nunique())


le = LabelEncoder()
df["mechanism_label"] = le.fit_transform(df["mechanism"])
print("Label encoding done. Example mapping:")
for mech, idx in zip(le.classes_[:10], le.transform(le.classes_[:10])):
    print(f"  {mech:25s} -> {idx}")
print("Total mechanisms:", len(le.classes_))

# Mechanism split (70/15/15)

mechs = np.array(sorted(df["mechanism_label"].unique()))
train_mechs, temp_mechs = train_test_split(mechs, test_size=0.30, random_state=RANDOM_SEED, shuffle=True)
val_mechs,   test_mechs = train_test_split(temp_mechs, test_size=0.50, random_state=RANDOM_SEED, shuffle=True)

train_df = df[df.mechanism_label.isin(train_mechs)].reset_index(drop=True)
val_df   = df[df.mechanism_label.isin(val_mechs)].reset_index(drop=True)
test_df  = df[df.mechanism_label.isin(test_mechs)].reset_index(drop=True)

print(f"Train/Val/Test mechanisms: {len(train_mechs)}/{len(val_mechs)}/{len(test_mechs)}")
print(f"Train/Val/Test samples: {len(train_df)}/{len(val_df)}/{len(test_df)}")

# 3) k-merisation -> tokenisation 
KMER = 5
MAX_KMER_SEQ = 512
PAD_ID = 0

def kmerize(seq, k=KMER):
    L = len(seq)
    if L < k: return []
    return [seq[i:i+k] for i in range(L - k + 1)]

# Compute k-mer lists
train_df["kmer_seq"] = train_df["sequence"].apply(kmerize)
val_df["kmer_seq"]   = val_df["sequence"].apply(kmerize)
test_df["kmer_seq"]  = test_df["sequence"].apply(kmerize)

# Build vocab from train set
kmer_counts = Counter(k for seq in train_df["kmer_seq"] for k in seq)
kmers_sorted = [k for k,_ in kmer_counts.most_common()]
kmer2id = {k: i+1 for i,k in enumerate(kmers_sorted)}  # 0 reserved for PAD
vocab_size = len(kmer2id) + 1
print("Built k-mer vocab (train-only). Vocab size:", vocab_size)

def encode_kmers(kmer_list, max_len=MAX_KMER_SEQ):
    ids = [kmer2id.get(k, PAD_ID) for k in kmer_list]
    if len(ids) < max_len:
        ids += [PAD_ID] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return np.array(ids, dtype=np.int64)

train_df["tok_seq"] = train_df["kmer_seq"].apply(lambda s: encode_kmers(s, max_len=MAX_KMER_SEQ))
val_df["tok_seq"]   = val_df["kmer_seq"].apply(lambda s: encode_kmers(s, max_len=MAX_KMER_SEQ))
test_df["tok_seq"]  = test_df["kmer_seq"].apply(lambda s: encode_kmers(s, max_len=MAX_KMER_SEQ))

def build_matrix_tok(dfp):
    X_tok = np.stack(dfp["tok_seq"].to_numpy())
    X_num = dfp[["gc_content","seq_len"]].to_numpy(np.float32)
    y = dfp["mechanism_label"].to_numpy(np.int64)
    return X_tok, X_num, y

Xtr_tok, Xtr_num, ytr = build_matrix_tok(train_df)
Xva_tok, Xva_num, yva = build_matrix_tok(val_df)
Xte_tok, Xte_num, yte = build_matrix_tok(test_df)
print("Tokenised shapes:", Xtr_tok.shape, Xtr_num.shape)

def mech_index(y):
    d = defaultdict(list)
    for i, lab in enumerate(y): d[lab].append(i)
    return {k: np.asarray(v, dtype=int) for k,v in d.items()}

idx_tr, idx_va, idx_te = mech_index(ytr), mech_index(yva), mech_index(yte)

def create_tasks_tok(X_tok, X_num, y, idx_map, num_tasks=1000, N=3, K=3, Q=5, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    tasks = []
    valid = [m for m, ids in idx_map.items() if len(ids) >= K + Q]
    if len(valid) < N:
        return tasks
    for _ in range(num_tasks):
        chosen_mechs = rng.choice(valid, size=N, replace=False)
        s_tok, s_num, s_y = [], [], []
        q_tok, q_num, q_y = [], [], []
        for j, m in enumerate(chosen_mechs):
            ids = rng.choice(idx_map[m], size=K + Q, replace=False)
            s_ids, q_ids = ids[:K], ids[K:K+Q]
            s_tok.append(X_tok[s_ids]); s_num.append(X_num[s_ids]); s_y.append(np.full(K, j, np.int64))
            q_tok.append(X_tok[q_ids]); q_num.append(X_num[q_ids]); q_y.append(np.full(Q, j, np.int64))
        tasks.append({
            "support_tok": np.vstack(s_tok),
            "support_num": np.vstack(s_num),
            "support_y": np.concatenate(s_y),
            "query_tok": np.vstack(q_tok),
            "query_num": np.vstack(q_num),
            "query_y": np.concatenate(q_y),
            "mechs": list(chosen_mechs)
        })
    return tasks

N, K, Q = 3, 3, 5
train_tasks = create_tasks_tok(Xtr_tok, Xtr_num, ytr, idx_tr, num_tasks=1500, N=N, K=K, Q=Q)
val_tasks   = create_tasks_tok(Xva_tok, Xva_num, yva, idx_va, num_tasks=300,  N=N, K=K, Q=Q)
test_tasks  = create_tasks_tok(Xte_tok, Xte_num, yte, idx_te, num_tasks=500,  N=N, K=K, Q=Q)
print(f"Tasks | train:{len(train_tasks)} val:{len(val_tasks)} test:{len(test_tasks)}  (N={N},K={K},Q={Q})")

class ProtoNet(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_features=2, use_cosine=True, pad_idx=0):
        super().__init__()
        self.use_cosine = use_cosine
        self.log_temp = nn.Parameter(torch.zeros(1))

        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=pad_idx)

        self.conv = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Sequential(
            nn.Linear(256 + num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, embed_dim)
        )

    def forward(self, X_tok, X_num):
        x = self.embedding(X_tok)
        x = self.conv(x.permute(0,2,1)).permute(0,2,1)
        x = self.transformer(x)
        x_pool = x.mean(dim=1)
        z = torch.cat([x_pool, X_num], dim=1)
        z = self.fc(z)
        if self.use_cosine:
            z = F.normalize(z, p=2, dim=1)
        return z

def compute_prototypes(emb, y_idx):
    classes = torch.unique(y_idx)
    protos = torch.stack([emb[y_idx==c].mean(0) for c in classes], dim=0)
    return protos, classes

def proto_loss(model, protos, q_emb, q_y, classes):
    if model.use_cosine:
        temp = torch.exp(model.log_temp)
        logits = (q_emb @ protos.T) / temp
        log_p = F.log_softmax(logits, dim=1)
    else:
        d = torch.cdist(q_emb, protos)
        log_p = F.log_softmax(-d, dim=1)
    return F.nll_loss(log_p, q_y)

@torch.no_grad()
def evaluate(model, tasks, device):
    model.eval()
    accs, losses = [], []
    for t in tasks:
        s_tok = torch.from_numpy(t["support_tok"]).long().to(device)
        q_tok = torch.from_numpy(t["query_tok"]).long().to(device)
        s_num = torch.from_numpy(t["support_num"]).float().to(device)
        q_num = torch.from_numpy(t["query_num"]).float().to(device)
        sY = torch.from_numpy(t["support_y"]).long().to(device)
        qY = torch.from_numpy(t["query_y"]).long().to(device)

        sZ, qZ = model(s_tok, s_num), model(q_tok, q_num)
        protos, classes = compute_prototypes(sZ, sY)
        if model.use_cosine:
            preds = (qZ @ protos.T).argmax(1)
        else:
            preds = (-torch.cdist(qZ, protos)).argmax(1)
        loss = proto_loss(model, protos, qZ, qY, classes).item()
        accs.append((preds==qY).float().mean().item())
        losses.append(loss)
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(losses))

model = ProtoNet(vocab_size=vocab_size, embed_dim=256, num_features=Xtr_num.shape[1], use_cosine=True, pad_idx=PAD_ID).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)

EPISODES = 1000
EVAL_EVERY = 25
PATIENCE = 8

best_val, bad = 0.0, 0
train_losses, train_accs, val_accs, val_losses = [], [], [], []

print("Training...")
for ep in tqdm(range(1, EPISODES+1)):
    model.train()
    t = random.choice(train_tasks)
    s_tok = torch.from_numpy(t["support_tok"]).long().to(device)
    q_tok = torch.from_numpy(t["query_tok"]).long().to(device)
    s_num = torch.from_numpy(t["support_num"]).float().to(device)
    q_num = torch.from_numpy(t["query_num"]).float().to(device)
    sY = torch.from_numpy(t["support_y"]).long().to(device)
    qY = torch.from_numpy(t["query_y"]).long().to(device)

    opt.zero_grad()
    sZ = model(s_tok, s_num)
    qZ = model(q_tok, q_num)
    protos, classes = compute_prototypes(sZ, sY)
    loss = proto_loss(model, protos, qZ, qY, classes)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step(); sched.step()
    train_losses.append(loss.item())

    if ep % EVAL_EVERY == 0:
        tr_acc, tr_std, _ = evaluate(model, train_tasks[:50], device)
        va_acc, va_std, va_loss = evaluate(model, val_tasks, device)
        train_accs.append(tr_acc); val_accs.append(va_acc); val_losses.append(va_loss)
        print(f"[{ep}/{EPISODES}] loss={np.mean(train_losses[-EVAL_EVERY:]):.3f}  "
              f"train={tr_acc:.3f}±{tr_std:.3f}  val={va_acc:.3f}±{va_std:.3f}  T={float(torch.exp(model.log_temp)):.3f}")
        if va_acc > best_val:
            best_val, bad = va_acc, 0
            torch.save(model.state_dict(), "best_protonet_tok.pt")
        else:
            bad += 1
        if bad >= PATIENCE:
            print("Early stopping.")
            break

print("Best Val Acc:", round(best_val, 4))

model.load_state_dict(torch.load("best_protonet_tok.pt", map_location=device))
test_acc, test_std, test_loss = evaluate(model, test_tasks, device)
print(f"TEST  acc={test_acc:.3f} ± {test_std:.3f} | loss={test_loss:.3f} | episodes={len(test_tasks)}")
print(f"Random baseline (1/N): {1.0/float(N):.3f}")

fig, axes = plt.subplots(1,2, figsize=(13,5))
axes[0].plot(train_losses, alpha=0.3)
axes[0].plot(pd.Series(train_losses).rolling(window=20, min_periods=1).mean(), label="smoothed")
axes[0].set_title("Training Loss"); axes[0].set_xlabel("Episode"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

eval_steps = [i*EVAL_EVERY for i in range(1, len(val_accs)+1)]
axes[1].plot(eval_steps, train_accs, marker='o', label="Train Acc")
axes[1].plot(eval_steps, val_accs, marker='s', label="Val Acc")
axes[1].axhline(1.0/N, color='r', linestyle='--', label=f"Random {1.0/N:.3f}")
axes[1].axhline(best_val, color='purple', linestyle=':', label=f"Best Val {best_val:.3f}")
axes[1].set_title("Accuracies"); axes[1].set_xlabel("Episode"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

#t-SNE
from sklearn.manifold import TSNE

SAMPLE_N = 2000
rng = np.random.default_rng(RANDOM_SEED)

n_total = len(y_vis)
idx = rng.choice(np.arange(n_total), size=min(SAMPLE_N, n_total), replace=False)
X_tok_vis_s = X_tok_vis[idx]
X_num_vis_s = X_num_vis[idx]
y_vis_s = np.array(y_vis)[idx]

print(f"Computing embeddings for t-SNE (sample size): {len(y_vis_s)}")
embeddings_vis = embed_sequences(meta_model, X_tok_vis_s, X_num_vis_s, batch_size=256)

print("Running t-SNE (may take a minute)...")
tsne = TSNE(
    n_components=2,
    init='pca',
    learning_rate='auto',
    random_state=RANDOM_SEED,
    perplexity=30
)
z2 = tsne.fit_transform(embeddings_vis)

plt.figure(figsize=(8,6))
labels, uniques = pd.factorize(y_vis_s)
scatter = plt.scatter(z2[:,0], z2[:,1], c=labels, s=10, cmap='tab20', alpha=0.8)
plt.title("t-SNE projection of learned embeddings (sampled subset)")
plt.xticks([]); plt.yticks([])

unique_mechs = le.inverse_transform(np.unique(y_vis_s))
handles, _ = scatter.legend_elements(num=len(unique_mechs))
max_labels = min(len(unique_mechs), len(handles))
plt.legend(handles[:max_labels], unique_mechs[:max_labels],
           bbox_to_anchor=(1.05, 1), loc="upper left",
           fontsize="small", title="Mechanism")
plt.tight_layout()
plt.show()