from pathlib import Path
from collections import Counter
import numpy as np

# 1. DATA LOADING & PREPROCESSING

DATA_PATH = Path(__file__).parent / "data" / "names.txt"


def load_names(path):
    names = path.read_text(encoding="utf-8-sig").splitlines()
    return [name.strip().lower() for name in names if name.strip()]


def split_data(names, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(names))

    n = len(names)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_names = [names[i] for i in indices[:train_end]]
    val_names = [names[i] for i in indices[train_end:val_end]]
    test_names = [names[i] for i in indices[val_end:]]

    return train_names, val_names, test_names


def build_vocab(names):
    return sorted(set("".join(names)) | {'.'})


def get_bigrams(names):
    bigrams = []

    for name in names:
        appended = f".{name}."

        for i in range(len(appended) - 1):
            bigrams.append(appended[i:i+2])

    return bigrams


# 2. BIGRAM MODEL

def bigram_prob(current_char, next_char, counts, vocab_size, alpha=1.0):
    total = 0

    for bigram, count in counts.items():
        if bigram[0] == current_char:
            total += count

    return (counts[f"{current_char}{next_char}"] + alpha) / (total + alpha * vocab_size)


def build_probability_matrix(vocab, counts, alpha):
    P = np.zeros((len(vocab), len(vocab)))

    for i, current_char in enumerate(vocab):
        for j, next_char in enumerate(vocab):
            P[i, j] = bigram_prob(current_char, next_char, counts, len(vocab), alpha)

    return P

# 3. TEXT GENERATION

def generate_name(P, vocab, stoi, rng, max_length=30):
    word = "."

    while len(word) < max_length:
        row = P[stoi[word[-1]]]
        next_char = rng.choice(vocab, p=row)

        word += next_char

        if next_char == ".":
            break

    return word

# 4. EVALUATION

def evaluate_nll(names, P, stoi):
    bigrams = get_bigrams(names)
    # probs = []
    # for bigram in bigrams:
    #     first = bigram[0]
    #     second = bigram[1]
    #     probs.append(P[stoi[first], stoi[second]])

    probs = [P[stoi[bigram[0]], stoi[bigram[1]]] for bigram in bigrams]
    nll = -np.mean(np.log(probs))

    return nll

def perplexity(nll):
    return np.exp(nll)

# 5. MAIN

def main():
    # Load and split data
    names = load_names(DATA_PATH)
    train_names, val_names, test_names = split_data(names)

    # Build vocabulary
    vocab = build_vocab(names)
    stoi = {char: i for i, char in enumerate(vocab)}
    alphas = [0.01, 0.1, 1, 10, 100, 1000]


    # Train bigram model
    counts = Counter(get_bigrams(train_names))
    P_different_alphas = [build_probability_matrix(vocab, counts, alpha) for alpha in alphas]
    # P = build_probability_matrix(vocab, counts, 0.01)

    # Print dataset statistics
    print(f"Total names: {len(names)}")
    print(f"Training: {len(train_names)}")
    print(f"Validation: {len(val_names)}")
    print(f"Test: {len(test_names)}")
    print(f"Vocabulary size: {len(vocab)}")

    # Generate names
    print("\nGenerated names:")
    rng = np.random.default_rng(123)

    for _ in range(20):
        print(generate_name(P_different_alphas[0], vocab, stoi, rng))


    best_nll = np.inf
    best_alpha = None
    #Evaluate model (after implementing evaluate_nll)
    for alpha, P in zip(alphas, P_different_alphas):
        train_nll = evaluate_nll(train_names, P, stoi)
        val_nll = evaluate_nll(val_names, P, stoi)
        train_ppl = perplexity(train_nll)
        val_ppl = perplexity(val_nll)

        print(f"Alpha = {alpha}")
        print(f"  Train NLL: {train_nll:.6f}")
        print(f"  Val NLL:   {val_nll:.6f}")
        print(f"  Train PPL: {train_ppl:.6f}")
        print(f"  Val PPL:   {val_ppl:.6f}")

        if val_nll < best_nll:
            best_nll = val_nll
            best_alpha = alpha

    print(f"Best alpha: {best_alpha} with NLL: {best_nll}")

    best_P = build_probability_matrix(vocab, counts, best_alpha)

    test_nll = evaluate_nll(test_names, best_P, stoi)
    test_ppl = perplexity(test_nll)

    print("\n=== FINAL TEST RESULTS ===")
    print(f"Selected alpha: {best_alpha}")
    print(f"Test NLL: {test_nll:.6f}")
    print(f"Test PPL: {test_ppl:.6f}")

if __name__ == "__main__":
    main()