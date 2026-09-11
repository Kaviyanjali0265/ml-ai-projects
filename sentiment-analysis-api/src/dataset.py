from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_CHECKPOINT = "distilbert-base-uncased"
MAX_LENGTH = 64
TRAIN_SIZE = 200
VAL_SIZE = 50
TEST_SIZE = 50


def load_imdb(train_size: int = TRAIN_SIZE, val_size: int = VAL_SIZE, test_size: int = TEST_SIZE):
    dataset = load_dataset("stanfordnlp/imdb")

    train = dataset["train"].shuffle(seed=42).select(range(train_size))
    test_full = dataset["test"].shuffle(seed=42)
    val = test_full.select(range(val_size))
    test = test_full.select(range(val_size, val_size + test_size))

    return train, val, test


def tokenize_dataset(train, val, test, checkpoint: str = MODEL_CHECKPOINT):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

    train = train.map(tokenize, batched=True)
    val = val.map(tokenize, batched=True)
    test = test.map(tokenize, batched=True)

    for ds in [train, val, test]:
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return train, val, test, tokenizer
