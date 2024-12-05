'''
Assignment 3
STEFANO ANDREOTTI
'''
import torch
from datasets import load_dataset
from collections import Counter
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Set the seed
seed = 42
torch.manual_seed(seed)
# Probably, this below must be changed if you work with a M1/M2/M3 Mac
torch.cuda.manual_seed(seed) # for CUDA
torch.backends.cudnn.deterministic = True # for CUDNN
torch.backends.benchmark = False # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.


# Question 6
def collate_fn(batch, pad_value):
    """
    Separate data (x) and target (y) pairs from the batch
    """
    data, targets = zip(*batch)

    padded_data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=pad_value)
    padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_value)

    return padded_data, padded_targets

def random_sample_next(model, x, prev_state, topk=None):
    """
    Randomly samples the next word based on the probability distribution.
    """
    # Perform forward-prop and get the output of the last time-step
    out, state = model(x, prev_state)
    last_out = out[0, -1, :]  # Vocabulary values of last element of sequence

    # Get the top-k indexes and their values
    topk = topk if topk else last_out.shape[0]

    top_logit, top_ix = torch.topk(last_out, k=topk, dim=-1)

    # Convert logits to probabilities and sample
    p = F.softmax(top_logit.detach(), dim=-1).cpu().numpy()  # Move to CPU before converting to numpy
    top_ix = top_ix.cpu().numpy()  # Move to CPU before converting to numpy

    # Check if top_ix is empty
    if len(top_ix) == 0:
        raise ValueError("No valid predictions were made (top_ix is empty).")

    sampled_ix = np.random.choice(top_ix, p=p)

    return sampled_ix, state

def sample_argmax(model, x, prev_state):
    """
    Samples the next word by picking the one with the highest probability (argmax strategy).
    """
    # Perform forward-prop and get the output of the last time-step
    out, state = model(x, prev_state)
    last_out = out[0, -1, :]  # Vocabulary values of last element of sequence

    # Get the index with the highest probability
    sampled_ix = torch.argmax(last_out).item()

    return sampled_ix, state

def sample(model, seed, stop_on, strategy="random", topk=5, max_seqlen=18):
    """
    Generates a sequence using the model.
    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps'
    if torch.backends.mps.is_available() else 'cpu')

    seed = seed if isinstance(seed, (list, tuple)) else [seed]
    model.eval()
    with torch.no_grad():
        sampled_ix_list = seed[:]
        x = torch.tensor([seed], device=DEVICE)

        prev_state = model.init_state(b_size=x.shape[0])

        # in LSTM prev_state is a tuple
        prev_state = tuple(s.to(DEVICE) for s in prev_state)

        for _ in range(max_seqlen - len(seed)):
            # Repeatedly predicts the next word/token based on the input sequence
            if strategy == "random":
                sampled_ix, prev_state = random_sample_next(model, x, prev_state, topk)
            elif strategy == "argmax":
                sampled_ix, prev_state = sample_argmax(model, x, prev_state)
            else:
                raise ValueError(f"Invalid sampling strategy: {strategy}")

            # The predicted token is appended to the sequence
            sampled_ix_list.append(sampled_ix)

            # The new token is used as the input for the next prediction
            x = torch.tensor([[sampled_ix]], device=DEVICE)

            # If the predicted token is word_to_int["<EOS>"] the function terminates the loop
            if sampled_ix == stop_on:
                break

    model.train()
    return sampled_ix_list

def train(model, data, num_epochs, criterion, lr=0.001, print_every=2, clip=None):
    """
    Function to train the model.
    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(DEVICE)
    criterion = criterion.to(DEVICE)
    model.train()

    loss_hist = []
    perplexity_hist = []
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_batches = len(data)
    epoch = 0

    generated_list = []
    while epoch < num_epochs:
        epoch += 1
        epoch_loss = 0.0

        for batch_idx, (x, y) in enumerate(data, 1):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()

            # Initialize hidden state
            prev_state = model.init_state(b_size=x.shape[0])
            prev_state = tuple(s.to(DEVICE) for s in prev_state)

            # Forward pass
            out, state = model(x, prev_state=prev_state)

            # Reshape output for CrossEntropyLoss [batch_size, vocab_size, sequence_length]
            loss_out = out.permute(0, 2, 1)

            # Calculate loss
            loss = criterion(loss_out, y)
            epoch_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        # Calculate average loss
        avg_epoch_loss = epoch_loss / total_batches
        loss_hist.append(avg_epoch_loss)

        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(avg_epoch_loss))
        perplexity_hist.append(perplexity.item())

        generated = sample(model, seed, word_to_int["<EOS>"], "argmax")
        generated = " ".join(keys_to_values(generated, int_to_word))

        generated_list.append(generated)

        print(f"Epoch: {epoch}/{num_epochs}, Loss: {avg_epoch_loss:8.4f}, Perplexity: {perplexity:8.4f}")
        if epoch == 1 or (print_every and (epoch % print_every) == 0):
            print(f"Generated text: {generated}\n")

        # Early stopping check
        if avg_epoch_loss < 1.5:
            print(f"\nTarget loss of 1.5 reached at epoch {epoch}!")
            break

    if len(generated_list) >= 3:
        print("Generated after first epoch:", generated_list[0])
        middle_index = len(generated_list) // 2
        print("Generated after middle epoch:", generated_list[middle_index])
        print("Generated after last epoch:", generated_list[-1])
    elif len(generated_list) == 2:
        print("Generated after first epoch:", generated_list[0])
        print("Generated after last epoch:", generated_list[-1])
    else:
        print("Generated after first epoch:", generated_list[0])

    return model, loss_hist, perplexity_hist

def tbtt_train(model, data, num_epochs, criterion, truncation_length=50, lr=0.001, print_every=2, clip=None):
    """
    Function to train the model with truncated backpropagation.
    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(DEVICE)
    criterion = criterion.to(DEVICE)
    model.train()

    loss_hist = []
    perplexity_hist = []
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_batches = len(data)
    epoch = 0

    generated_list = []

    while epoch < num_epochs:
        epoch += 1
        epoch_loss = 0.0

        for batch_idx, (x, y) in enumerate(data, 1):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            # Initialize hidden state
            prev_state = model.init_state(b_size=x.shape[0])
            prev_state = tuple(s.to(DEVICE) for s in prev_state)

            for i in range(0, x.size(1), truncation_length):  # Truncated loop
                x_truncated = x[:, i:i + truncation_length]
                y_truncated = y[:, i:i + truncation_length]

                # Forward pass
                out, state = model(x_truncated, prev_state=prev_state)
                prev_state = tuple(s.detach() for s in state)  # Detach each element in the tuple

                # Reshape output for CrossEntropyLoss
                loss_out = out.permute(0, 2, 1)

                # Calculate loss
                loss = criterion(loss_out, y_truncated)
                epoch_loss += loss.item()

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                if clip:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / total_batches
        loss_hist.append(avg_epoch_loss)

        # Calculate perplexity directly from cross-entropy loss
        perplexity = torch.exp(torch.tensor(avg_epoch_loss))
        perplexity_hist.append(perplexity.item())

        generated = sample(model, seed, word_to_int["<EOS>"], "argmax")
        generated = " ".join(keys_to_values(generated, int_to_word))

        generated_list.append(generated)

        print(f"Epoch: {epoch}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, Perplexity: {perplexity:.4f}")
        if epoch == 1 or (print_every and (epoch % print_every) == 0):
            print(f"Generated text: {generated}\n")

        # Early stopping condition
        if avg_epoch_loss < 1.5:
            print(f"Target loss of 1.5 reached at epoch {epoch}!")
            break

    if len(generated_list) >= 3:
        print("Generated after first epoch:", generated_list[0])
        middle_index = len(generated_list) // 2
        print("Generated after middle epoch:", generated_list[middle_index])
        print("Generated after last epoch:", generated_list[-1])
    elif len(generated_list) == 2:
        print("Generated after first epoch:", generated_list[0])
        print("Generated after last epoch:", generated_list[-1])
    else:
        print("Generated after first epoch:", generated_list[0])

    return model, loss_hist, perplexity_hist

def tokenize_and_map(sentence, word_to_int):
    """
    Function to tokenize and map the input sentence to words.
    """
    tokens = sentence.split(" ")  # Split sentence into words
    return [word_to_int[word] for word in tokens if word in word_to_int]

def decode_sequence(generated_tokens, int_to_word):
    """
    Function to decode the generated sequence.
    """
    return [int_to_word[token] for token in generated_tokens if token in int_to_word]

def keys_to_values(keys, map, default_if_missing=None):
    return [map.get(key, default_if_missing) for key in keys]

def make_plots(loss_hist, perplexity_hist):
    """
    Function to make plots for loss and perplexity.
    """
    # Plot loss history
    plt.plot(loss_hist, label='Loss History', color='blue')
    plt.axhline(y=1.5, color='red', linestyle='--', label='Threshold (1.5)')
    plt.title('Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot perplexity history
    plt.plot(perplexity_hist, label='Perplexity History', color='green')
    plt.title('Perplexity History')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    '''
    Data
    '''
    # Question 1
    ds = load_dataset("heegyu/news-category-dataset")
    print(type(ds))
    print(ds['train'])
    print(ds['train'][0])


    # Question 2
    ds = [news['headline'] for news in ds['train'] if news['category'] == 'POLITICS']
    print("First headline: ", ds[0])
    print(len(ds))


    # Question 3
    ds = [headline.lower() for headline in ds]
    ds = [headline.split(" ") for headline in ds]

    # Add <EOS> at the end of every headline
    for headline in ds:
        headline.append('<EOS>')


    # Question 4
    all_words = [word for headline in ds for word in headline]

    # Count word frequencies
    word_counts = Counter(all_words)
    most_common_words = word_counts.most_common(6)
    print("5 most common words:", most_common_words)

    # Extract unique words
    unique_words = set(word for headline in ds for word in headline)

    # Create vocabulary with <EOS> at the beginning and PAD at the end
    unique_words = {word for word in unique_words if word and word not in ["<EOS>", "PAD"]}
    # Sorting of unique_words
    word_vocab = ["<EOS>"] + sorted(list(unique_words)) + ["PAD"]
    total_words = len(word_vocab) - 2
    print("Total number of words in vocabulary (no <EOS> and PAD):", total_words)

    # Dictionary representing a mapping from words of our word_vocab to integer values
    word_to_int = {word: i for i, word in enumerate(word_vocab)}

    # Dictionary representing the inverse of `word_to_int`
    int_to_word = {word:i for i, word in word_to_int.items()}


    # Question 5
    class Dataset:
        def __init__(self, sequences, word_to_int):
            self.sequences = sequences
            self.word_to_int = word_to_int

            # Convert each sequence (list of words) to indexes using map
            self.indexed_sequences = [
                [self.word_to_int[word] for word in sequence if word in self.word_to_int]
                for sequence in self.sequences
            ] # the problem is that if in the sequence there is a word (ex '') without mapping, skip it

        def __getitem__(self, idx):
            # Get the indexed sequence at the given index
            indexed_seq = self.indexed_sequences[idx]

            # Create x (all indexes except the last one) and y (all indexes except the first one)
            x = indexed_seq[:-1]
            y = indexed_seq[1:]

            return torch.tensor(x), torch.tensor(y)

        def __len__(self):
            # Return the total number of sequences
            return len(self.indexed_sequences)


    # Question 6
    batch_size = 8
    dataset = Dataset(ds, word_to_int)

    if batch_size == 1:
      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda batch: collate_fn(batch, word_to_int["PAD"]))


    '''
    Model
    '''
    class LSTMModel(nn.Module):
        def __init__(self, map, hidden_size=1024, emb_dim=150, n_layers=1):
            super(LSTMModel, self).__init__()

            self.vocab_size  = len(map)
            self.hidden_size = hidden_size
            self.emb_dim     = emb_dim
            self.n_layers    = n_layers

            # Embedding layer
            self.embedding = nn.Embedding(
                num_embeddings=self.vocab_size,
                embedding_dim=self.emb_dim,
                padding_idx=map["PAD"]
            )

            # LSTM layer with potential stacking
            self.lstm = nn.LSTM(
                input_size=self.emb_dim,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                batch_first=True
            )

            # Fully connected layer to project LSTM outputs to vocabulary size
            self.fc = nn.Linear(
                in_features=self.hidden_size,
                out_features=self.vocab_size
            )

        def forward(self, x, prev_state):
            # Embedding lookup for input tokens
            embed = self.embedding(x)

            # Pass embeddings through the LSTM
            yhat, state = self.lstm(embed, prev_state)  # yhat: (batch, seq_length, hidden_size)

            # Pass through the fully connected layer to get logits
            out = self.fc(yhat)

            return out, state

        def init_state(self, b_size=1):
            # Initializes hidden and cell states with zeros
            return (torch.zeros(self.n_layers, b_size, self.hidden_size),
                    torch.zeros(self.n_layers, b_size, self.hidden_size))


    '''
    Evaluation, part 1
    '''
    model = LSTMModel(word_to_int)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(DEVICE)
    print("Working on", DEVICE)

    # Start with any prompt and generate three sentences with the sampling strategy.
    seed = "the president wants"
    seed = tokenize_and_map(seed, word_to_int)
    print("Random")
    print(f"Seed: {seed}")

    for i in range(3):
        generated = sample(model, seed, word_to_int["<EOS>"], "random")
        generated = decode_sequence(generated, int_to_word)
        print("Generated: ", ' '.join(str(e) for e in generated))

    # Start with any prompt and generate three sentences with the sampling strategy.
    seed = "the president wants"
    seed = tokenize_and_map(seed, word_to_int)  # Convert to token indices
    print("\nGreedy")
    print(f"Seed: {seed}")

    for i in range(3):
        generated = sample(model, seed, word_to_int["<EOS>"], "argmax")
        generated = decode_sequence(generated, int_to_word)
        print("Generated: ", ' '.join(str(e) for e in generated))


    '''
    Training
    '''
    epochs = 12
    criterion = nn.CrossEntropyLoss(ignore_index=word_to_int["PAD"])
    model, loss_hist, perplexity_hist = train(model=model, data=dataloader, num_epochs=epochs, criterion=criterion,
                                              lr=1e-3, print_every=3, clip=1)
    # Plot loss and perplexity
    make_plots(loss_hist, perplexity_hist)

    truncation_len = 50
    epochs = 5
    hidden_size = 2048
    model = LSTMModel(word_to_int, hidden_size)
    model, loss_hist, perplexity_hist = tbtt_train(model=model, data=dataloader, num_epochs=epochs, criterion=criterion,
                                                   truncation_length=truncation_len, lr=1e-3, print_every=2, clip=1)
    # Plot loss and perplexity
    make_plots(loss_hist, perplexity_hist)


    '''
    Evaluation, part 2
    '''
    seed = "the president wants"
    seed = tokenize_and_map(seed, word_to_int)  # Convert to token indices
    print(f"Seed: {seed}")

    for i in range(3):
        generated = sample(model, seed, word_to_int["<EOS>"], "random")
        generated = " ".join(keys_to_values(generated, int_to_word))
        print("Generated: ", generated)
    print()

    for i in range(3):
        generated = sample(model, seed, word_to_int["<EOS>"], "argmax")
        generated = " ".join(keys_to_values(generated, int_to_word))
        print("Generated: ", generated)
