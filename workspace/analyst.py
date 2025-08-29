import pandas as pd
import matplotlib.pyplot as plt

path = "data/raw/function.json"
df = pd.read_json(path)

df["tokens"] = df["func"].map(lambda x: len(x.split()) if isinstance(x, str) else 0)
df["k_hundred"] = df["tokens"].apply(lambda n: 0 if n == 0 else (n - 1)//100 + 1)

cap_k = 22
df["k_bucket"] = df["k_hundred"].where(df["k_hundred"] <= cap_k, other=cap_k + 1)
counts = df.groupby("k_bucket").size().sort_index()

total = len(df)
max_tokens = int(df["tokens"].max())
print(f"Total functions: {total}")
print(f"Max tokens: {max_tokens}")
print("Distribution by hundred-level:")

values = []
percents = []
for k, countFunc in counts.items():
    if k == 0:
        print(f"- 0 tokens: {countFunc}")
    elif k == cap_k + 1:
        lo = cap_k * 100 + 1
        print(f"- {cap_k}k+ (≥{lo} tokens): {countFunc}")
    else:
        lo = (k - 1) * 100 + 1
        hi = k * 100
        print(f"- {k}k ({lo}-{hi} tokens): {countFunc}")
    values.append(int(countFunc))
    percents.append((countFunc / total) * 100 if total > 0 else 0.0)

# Vẽ biểu đồ cột và hiển thị phần trăm
plt.figure(figsize=(12, 6))
bars = plt.bar(counts.index, counts.values, color="#4C78A8", edgecolor="black")
plt.title("Function token distribution (by hundred-level)")
plt.xlabel("Token bucket")
plt.ylabel("Count")

# Tạo custom labels cho trục x
labels = []
for k in counts.index:
    if k == 0:
        labels.append("0")
    elif k == cap_k + 1:
        labels.append(f"{cap_k*100}+")
    else:
        labels.append(f"{k*100}")

plt.xticks(counts.index, labels)

# Hiển thị % trên đầu cột
for bar, pct in zip(bars, percents):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.savefig("workspace/function_token_distribution.png", dpi=150)
plt.show()