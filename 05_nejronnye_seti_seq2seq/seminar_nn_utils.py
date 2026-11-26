"""Общий код семинара «Нейронные сети: от нейрона до seq2seq с вниманием».

Тот же код показан в демонстрационных блокнотах seminar_nn_1, seminar_nn_2, seminar_nn_3.
Модуль нужен блокноту с задачами, чтобы не повторять определения.
"""
import json
import random
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
import torch.nn.functional as F

# <<style>>
SEED = 42
BLUE, ORANGE, MUTED, GRID = "#2a78d6", "#eb6834", "#6b7280", "#e5e7eb"
GREEN, VIOLET = "#1baf7a", "#6250d6"
# расходящаяся шкала: синий (класс 0), серый (0.5), оранжевый (класс 1)
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#cfe0f7", "#f0f0ee", "#f9d5c5", "#eb6834"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#f6f9fe", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "axes.titleweight": "bold"})


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
# <</style>>


# <<mlp>>
def make_mlp(n_hidden, n_layers=1, activation=nn.ReLU):
    # Сеть прямого распространения: 2 входа, n_layers скрытых слоёв по n_hidden нейронов, 2 выхода (оценки классов).
    # activation=None даёт сеть без нелинейности.
    layers, n_in = [], 2
    for _ in range(n_layers):
        layers.append(nn.Linear(n_in, n_hidden))
        if activation is not None:
            layers.append(activation())
        n_in = n_hidden
    layers.append(nn.Linear(n_in, 2))
    return nn.Sequential(*layers)


def train_mlp(model, X, y, lr=0.1, epochs=500, X_val=None, y_val=None, optimizer="sgd", weight_decay=0.0):
    # Обучение на всей выборке сразу. Возвращает историю потерь на обучении и на проверочной части.
    X, y = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)
    if X_val is not None:
        X_val, y_val = torch.as_tensor(X_val, dtype=torch.float32), torch.as_tensor(y_val, dtype=torch.long)
    opt = (torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay) if optimizer == "sgd"
           else torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay))
    history = {"train": [], "val": []}
    for _ in range(epochs):
        loss = F.cross_entropy(model(X), y)          # прямой проход и функция потерь
        opt.zero_grad()
        loss.backward()                              # обратный проход: градиенты по всем весам
        opt.step()                                   # шаг против градиента
        history["train"].append(loss.item())
        if X_val is not None:
            with torch.no_grad():
                history["val"].append(F.cross_entropy(model(X_val), y_val).item())
    return history


def mlp_accuracy(model, X, y):
    with torch.no_grad():
        pred = model(torch.as_tensor(X, dtype=torch.float32)).argmax(1).numpy()
    return float((pred == np.asarray(y)).mean())


def plot_mlp_boundary(ax, model, X, y, title="", lim=(-1.8, 2.8, -1.5, 2.0)):
    # Закрашивает плоскость вероятностью класса 1 и рисует границу, на которой она равна 0.5.
    gx, gy = np.meshgrid(np.linspace(lim[0], lim[1], 250), np.linspace(lim[2], lim[3], 250))
    grid = torch.tensor(np.c_[gx.ravel(), gy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        p = F.softmax(model(grid), dim=1)[:, 1].numpy().reshape(gx.shape)
    ax.contourf(gx, gy, p, levels=np.linspace(0, 1, 11), cmap=DIV, alpha=0.9)
    ax.contour(gx, gy, p, levels=[0.5], colors="#1f2430", linewidths=1.8)
    for cls, col in [(0, BLUE), (1, ORANGE)]:
        ax.scatter(X[y == cls, 0], X[y == cls, 1], s=16, color=col, edgecolor="white", linewidth=0.5)
    ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3]); ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
# <</mlp>>


# <<news>>
def find_data(name):
    # каталог data лежит в корне курса, он ищется вверх по дереву каталогов
    for parent in [Path.cwd()] + list(Path.cwd().parents):
        if (parent / "data" / name).exists():
            return parent / "data" / name
    raise FileNotFoundError(f"Файл data/{name} не найден. Выполните семинар 0: он загружает данные курса.")


def load_news():
    # Новости двух рубрик: 0 «Экономика», 1 «Наука и техника». Возвращает тексты и метки.
    rows = [json.loads(line) for line in open(find_data("lenta_ru_short.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r["topic"] in ("Экономика", "Наука и техника")]
    texts = [(r["title"] + ". " + r["text"]).replace("\xa0", " ") for r in rows]
    labels = np.array([1 if r["topic"] == "Наука и техника" else 0 for r in rows])
    return texts, labels


def load_titles():
    # Заголовки всех новостей файла в нижнем регистре: корпус для символьной языковой модели.
    rows = [json.loads(line) for line in open(find_data("lenta_ru_short.jsonl"), encoding="utf-8")]
    titles = [re.sub(r"\s+", " ", r["title"].replace("\xa0", " ").lower()).strip() for r in rows]
    return [t for t in titles if 15 <= len(t) <= 80]


TOKEN_RE = re.compile(r"[а-яёa-z]{2,}")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def build_vocab(token_lists, min_count=3):
    # Слово получает номер, если встретилось не менее min_count раз. Номер 0 отдан неизвестным словам.
    counts = {}
    for toks in token_lists:
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
    words = ["<unk>"] + sorted(w for w, c in counts.items() if c >= min_count)
    return {w: i for i, w in enumerate(words)}


def to_bag(token_lists, vocab):
    # Формат слоя nn.EmbeddingBag: все номера слов подряд и позиции, с которых начинается каждый текст.
    flat, offsets = [], []
    for toks in token_lists:
        offsets.append(len(flat))
        ids = [vocab.get(t, 0) for t in toks] or [0]
        flat.extend(ids)
    return torch.tensor(flat), torch.tensor(offsets)


class MeanEmbeddingClassifier(nn.Module):
    # Текст -> среднее эмбеддингов его слов -> скрытый слой с ReLU -> оценки двух классов.
    def __init__(self, vocab_size, emb_dim=32, hidden=32):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, emb_dim, mode="mean")
        self.hidden = nn.Linear(emb_dim, hidden)
        self.out = nn.Linear(hidden, 2)

    def forward(self, flat, offsets):
        x = self.emb(flat, offsets)                 # (число текстов, emb_dim)
        return self.out(torch.relu(self.hidden(x)))


def train_text_classifier(model, train, y_train, test, y_test, epochs=12, lr=0.01, batch_size=64, seed=SEED):
    # train и test: списки списков номеров слов. Возвращает историю потерь и доли верных ответов по эпохам.
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long)

    def bag(docs):
        flat, offsets = [], []
        for d in docs:
            offsets.append(len(flat)); flat.extend(d or [0])
        return torch.tensor(flat), torch.tensor(offsets)

    test_bag = bag(test)
    history = {"loss": [], "train_acc": [], "test_acc": []}
    for _ in range(epochs):
        order = np.random.permutation(len(train))
        total, correct = 0.0, 0
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            flat, offsets = bag([train[j] for j in idx])
            logits = model(flat, offsets)
            loss = F.cross_entropy(logits, y_train_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx); correct += (logits.argmax(1) == y_train_t[idx]).sum().item()
        with torch.no_grad():
            test_acc = (model(*test_bag).argmax(1).numpy() == np.asarray(y_test)).mean()
        history["loss"].append(total / len(train)); history["train_acc"].append(correct / len(train))
        history["test_acc"].append(float(test_acc))
    return history
# <</news>>


# <<charlm>>
PAD, BOS, EOS = 0, 1, 2      # служебные символы: заполнитель, начало и конец последовательности


class CharVocab:
    def __init__(self, texts):
        self.chars = ["<pad>", "<bos>", "<eos>"] + sorted({c for t in texts for c in t})
        self.index = {c: i for i, c in enumerate(self.chars)}

    def __len__(self):
        return len(self.chars)

    def encode(self, text):
        return [self.index[c] for c in text if c in self.index]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids if i > EOS)


def pad_batch(seqs):
    # Выравнивает последовательности номеров до общей длины заполнителем PAD.
    out = torch.full((len(seqs), max(len(s) for s in seqs)), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s)
    return out


class CharLM(nn.Module):
    # Символьная языковая модель: эмбеддинг символа -> рекуррентный слой -> оценки следующего символа.
    def __init__(self, vocab_size, emb_dim=32, hidden=128, cell="rnn"):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell](emb_dim, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x, state=None):
        h, state = self.rnn(self.emb(x), state)
        return self.out(h), state


def lm_batches(encoded, batch_size, shuffle=True):
    order = np.random.permutation(len(encoded)) if shuffle else np.arange(len(encoded))
    for i in range(0, len(order), batch_size):
        seqs = [[BOS] + encoded[j] + [EOS] for j in order[i:i + batch_size]]
        batch = pad_batch(seqs)
        yield batch[:, :-1], batch[:, 1:]            # вход и тот же текст со сдвигом на один символ


def lm_perplexity(model, encoded, batch_size=128):
    # Перплексия на символ: экспонента средней кросс-энтропии.
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in lm_batches(encoded, batch_size, shuffle=False):
            logits, _ = model(x)
            total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=PAD, reduction="sum").item()
            count += (y != PAD).sum().item()
    return float(np.exp(total / count))


def train_char_lm(model, train, valid, epochs=6, lr=3e-3, batch_size=64, seed=SEED, verbose=True):
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        for x, y in lm_batches(train, batch_size):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=PAD)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)      # ограничение нормы градиента
            opt.step()
        history.append(lm_perplexity(model, valid))
        if verbose:
            print(f"эпоха {epoch + 1}: перплексия на отложенных заголовках {history[-1]:.2f}")
    return history


def generate(model, vocab, prefix="", temperature=1.0, max_len=70, seed=None):
    # Порождение по одному символу: распределение следующего символа делится на температуру и из него берётся выборка.
    if seed is not None:
        torch.manual_seed(seed)
    ids = [BOS] + vocab.encode(prefix)
    x = torch.tensor([ids])
    out, state = model(x)
    result = list(ids[1:])
    for _ in range(max_len):
        probs = F.softmax(out[0, -1] / temperature, dim=0)
        nxt = int(torch.multinomial(probs, 1))
        if nxt == EOS:
            break
        result.append(nxt)
        out, state = model(torch.tensor([[nxt]]), state)
    return vocab.decode(result)
# <</charlm>>


# <<memory>>
N_CLASSES, N_FILL = 8, 8      # восемь значащих символов и восемь символов-заполнителей


def memory_batch(length, batch_size):
    # Первый символ случайный из N_CLASSES, остальные length-1 символов случайные заполнители.
    # Ответ: первый символ. Сеть должна пронести его через всю последовательность.
    first = torch.randint(0, N_CLASSES, (batch_size,))
    x = torch.randint(N_CLASSES, N_CLASSES + N_FILL, (batch_size, length))
    x[:, 0] = first
    return x, first


class SequenceClassifier(nn.Module):
    # Эмбеддинг символа -> рекуррентный слой -> класс по последнему скрытому состоянию.
    # cell: "rnn", "gru", "lstm" (инициализация PyTorch по умолчанию) или "lstm_open" (вентиль забывания открыт).
    def __init__(self, cell="rnn", emb_dim=16, hidden=32):
        super().__init__()
        self.emb = nn.Embedding(N_CLASSES + N_FILL, emb_dim)
        layer = {"rnn": nn.RNN, "lstm": nn.LSTM, "lstm_open": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = layer(emb_dim, hidden, batch_first=True)
        if cell == "lstm_open":
            open_forget_gate(self.rnn, hidden)
        self.out = nn.Linear(hidden, N_CLASSES)

    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.out(h[:, -1])


def open_forget_gate(lstm, hidden, value=3.0):
    # Смещение вентиля забывания равно +3: sigma(3) = 0.95, поэтому в начале обучения ячейка
    # переносит своё содержимое на следующий шаг почти без потерь, и градиент проходит вдоль ячейки.
    # В PyTorch смещения четырёх вентилей лежат подряд: вход, забывание, кандидат, выход.
    lstm.bias_ih_l0.data[hidden:2 * hidden] = value
    lstm.bias_hh_l0.data[hidden:2 * hidden] = 0.0


def train_memory(cell, length, steps=400, batch_size=64, lr=3e-3, seed=SEED):
    # Обучает модель на задаче «запомнить первый символ» и возвращает долю верных ответов на 2000 новых примерах.
    set_seed(seed)
    model = SequenceClassifier(cell)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        x, y = memory_batch(length, batch_size)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        x, y = memory_batch(length, 2000)
        return float((model(x).argmax(1) == y).float().mean())
# <</memory>>


# <<dates>>
MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
          "ноября", "декабря"]


def date_pair(rng):
    # Случайная дата в двух записях: «12 ноября 2026» и «2026-11-12».
    day, month, year = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1950, 2049)
    return f"{day} {MONTHS[month - 1]} {year}", f"{year}-{month:02d}-{day:02d}"


def reverse_pair(rng, min_len=3, max_len=20, length=None):
    # Случайная строка цифр и она же в обратном порядке.
    n = length or rng.randint(min_len, max_len)
    s = "".join(rng.choice("0123456789") for _ in range(n))
    return s, s[::-1]


def make_pairs(pair_fn, n, seed=SEED, **kw):
    rng = random.Random(seed)
    return [pair_fn(rng, **kw) for _ in range(n)]


def encode_pairs(pairs, vocab):
    # Возвращает вход кодировщика, длины входов, вход декодировщика (с BOS) и ожидаемый выход (с EOS).
    src = [vocab.encode(s) for s, _ in pairs]
    tgt = [vocab.encode(t) for _, t in pairs]
    return (pad_batch(src), torch.tensor([len(s) for s in src]),
            pad_batch([[BOS] + t for t in tgt]), pad_batch([t + [EOS] for t in tgt]))
# <</dates>>


# <<seq2seq>>
class Encoder(nn.Module):
    # Кодировщик: эмбеддинги символов входа -> GRU. Возвращает состояния на всех позициях и последнее состояние.
    def __init__(self, vocab_size, emb_dim=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.rnn = nn.GRU(emb_dim, hidden, batch_first=True)

    def forward(self, src, src_len):
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(src), src_len, batch_first=True, enforce_sorted=False)
        states, last = self.rnn(packed)
        states, _ = nn.utils.rnn.pad_packed_sequence(states, batch_first=True, total_length=src.size(1))
        return states, last                          # (B, S, H) и (1, B, H)


class Decoder(nn.Module):
    # Декодировщик на GRU. При attention=True на каждом шаге добавляется контекстный вектор:
    # взвешенная сумма состояний кодировщика с весами внимания.
    def __init__(self, vocab_size, emb_dim=32, hidden=64, attention=False):
        super().__init__()
        self.attention = attention
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.rnn = nn.GRU(emb_dim + (hidden if attention else 0), hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def step(self, prev_token, hidden, enc_states, src_mask):
        # prev_token: (B,) предыдущий символ, hidden: (1, B, H) состояние декодировщика
        x = self.emb(prev_token)                                              # (B, E)
        weights = None
        if self.attention:
            query = hidden[-1]                                                # (B, H) текущее состояние декодировщика
            scores = torch.bmm(enc_states, query.unsqueeze(2)).squeeze(2)     # (B, S) скалярные произведения
            scores = scores.masked_fill(~src_mask, float("-inf"))             # позиции заполнителя исключаются
            weights = F.softmax(scores, dim=1)                                # (B, S) веса внимания
            context = torch.bmm(weights.unsqueeze(1), enc_states).squeeze(1)  # (B, H) контекстный вектор
            x = torch.cat([x, context], dim=1)
        output, hidden = self.rnn(x.unsqueeze(1), hidden)
        return self.out(output.squeeze(1)), hidden, weights


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden=64, attention=False):
        super().__init__()
        self.encoder = Encoder(vocab_size, emb_dim, hidden)
        self.decoder = Decoder(vocab_size, emb_dim, hidden, attention)

    def forward(self, src, src_len, tgt_in):
        # Обучение с принудительной подачей правильного ответа (teacher forcing):
        # на вход шага t подаётся правильный символ t-1, собственные предсказания модели не используются.
        enc_states, hidden = self.encoder(src, src_len)
        mask = src != PAD
        logits = []
        for t in range(tgt_in.size(1)):
            step_logits, hidden, _ = self.decoder.step(tgt_in[:, t], hidden, enc_states, mask)
            logits.append(step_logits)
        return torch.stack(logits, dim=1)                                     # (B, T, размер словаря)

    @torch.no_grad()
    def greedy(self, src, src_len, max_len=25):
        # Жадный вывод: на каждом шаге берётся самый вероятный символ, и он же подаётся на следующий шаг.
        enc_states, hidden = self.encoder(src, src_len)
        mask = src != PAD
        token = torch.full((src.size(0),), BOS, dtype=torch.long)
        tokens, attn = [], []
        for _ in range(max_len):
            step_logits, hidden, weights = self.decoder.step(token, hidden, enc_states, mask)
            token = step_logits.argmax(1)
            tokens.append(token)
            if weights is not None:
                attn.append(weights)
        return torch.stack(tokens, dim=1), (torch.stack(attn, dim=1) if attn else None)


def translate(model, vocab, text, max_len=25):
    # Перевод одной строки. Возвращает строку и матрицу внимания (выходные символы x входные символы).
    src = torch.tensor([vocab.encode(text)])
    tokens, attn = model.greedy(src, torch.tensor([src.size(1)]), max_len)
    ids = tokens[0].tolist()
    n = ids.index(EOS) if EOS in ids else len(ids)
    return vocab.decode(ids[:n]), (attn[0, :n + 1].numpy() if attn is not None else None)


def exact_match(model, vocab, pairs, max_len=25):
    # Доля примеров, в которых выход совпал с ответом целиком.
    src, src_len, _, _ = encode_pairs(pairs, vocab)
    tokens, _ = model.greedy(src, src_len, max_len)
    hits = 0
    for row, (_, target) in zip(tokens.tolist(), pairs):
        n = row.index(EOS) if EOS in row else len(row)
        hits += vocab.decode(row[:n]) == target
    return hits / len(pairs)


def train_seq2seq(model, vocab, train_pairs, test_pairs, epochs=6, lr=3e-3, batch_size=64, seed=SEED, verbose=True):
    # Возвращает долю точных совпадений на тестовых парах после каждой эпохи.
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    max_len = max(len(t) for _, t in test_pairs) + 2
    history = []
    for epoch in range(epochs):
        order = np.random.permutation(len(train_pairs))
        for i in range(0, len(order), batch_size):
            src, src_len, tgt_in, tgt_out = encode_pairs([train_pairs[j] for j in order[i:i + batch_size]], vocab)
            logits = model(src, src_len, tgt_in)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1), ignore_index=PAD)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        history.append(exact_match(model, vocab, test_pairs, max_len))
        if verbose:
            print(f"эпоха {epoch + 1}: потери {loss.item():.3f}, точных совпадений на тесте {history[-1]:.3f}")
    return history


def plot_attention(ax, weights, source, output, title=""):
    # Карта внимания: строка соответствует выходному символу, столбец входному.
    ax.imshow(weights, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(source))); ax.set_xticklabels(["␣" if c == " " else c for c in source])
    labels = list(output) + ["конец"]
    ax.set_yticks(range(len(labels[:weights.shape[0]]))); ax.set_yticklabels(labels[:weights.shape[0]])
    ax.set_xlabel("вход"); ax.set_ylabel("выход"); ax.set_title(title)
    ax.spines[["left", "bottom"]].set_visible(False); ax.tick_params(length=0)
# <</seq2seq>>
