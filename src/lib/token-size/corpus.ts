/**
 * The sample text every byte-per-token measurement is taken over.
 *
 * A BPE tokenizer's ratio of bytes to tokens is not a constant: prose reads at
 * roughly three and a half to four bytes per token, while code, numbers, and
 * multibyte scripts run heavier. Sizing a token count against a fixed bundle
 * keeps every model on this page measured the same way, so the models are
 * comparable even though real text will drift from the figure.
 *
 * The bundle mixes the shapes a prompt and a dataset actually take: English
 * prose, function definitions in a few languages, punctuation, numbers, URLs,
 * and a paragraph of CJK so multibyte text pulls its weight.
 */

export const TOKENIZER_CORPUS = `The quick brown fox jumps over the lazy dog. This is a standard English pangram that exercises every letter of the alphabet at least once, and it is often used to test fonts and keyboards because it is short and memorable.

Large language models process text by splitting it into tokens, which are short chunks of words, subwords, punctuation, or individual characters. The number of tokens in a piece of text depends on the tokenizer the model was trained with, and different models use different vocabulary sizes and different tokenization strategies. For English text, a common rule of thumb is that one token is roughly equivalent to about three quarters of a word, or around four characters, but the exact ratio varies from model to model.

function fibonacci(n) { if (n < 2) return n; let a = 0, b = 1; for (let i = 2; i <= n; i++) { let c = a + b; a = b; b = c; } return b; }
const results = users.map(u => ({ id: u.id, name: u.name, score: computeScore(u) })).filter(r => r.score > 0.5).sort((a, b) => b.score - a.score);
def train(model, data, epochs=3):
    for epoch in range(epochs):
        for batch in data:
            loss = model(batch).loss
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch}: loss={loss.item():.4f}")

In the field of artificial intelligence, transformer models have become the dominant architecture for natural language processing tasks. An attention mechanism allows the network to weigh the importance of different input tokens when producing an output, which helps it capture long range dependencies in the text. Context windows have grown from a few thousand tokens to hundreds of thousands, and modern models can process entire books or long conversations in a single pass. The self-attention operation compares every token with every other token, producing a matrix of attention scores that scales quadratically with the sequence length, which is why researchers are constantly seeking more efficient variants such as linear attention or sliding window attention.

机器学习和人工智能正在快速发展。大语言模型需要大量的计算资源进行训练，包括数千块高端图形处理器。多语言模型能够处理中文、英文、日文、韩文以及其他多种语言的文本，这对于全球化的应用非常重要。训练数据通常包含来自互联网的大量文本，这些数据需要进行清洗和去重。

This is a test of repeated words and words that repeat, with numbers like 1234567890 and 3.14159265358979, plus symbols: @#$%^&*()_+ and URLs like https://example.com/path?query=1&x=2. Email addresses are common, along with dates like 2024-01-31 and times like 14:30:00. Quoted text "double quotes" and 'single quotes' appear throughout. The token count matters because providers price API calls per token, and a large context can quickly become expensive.

`

/** The byte length of the corpus, computed once. */
export const TOKENIZER_CORPUS_BYTES = new TextEncoder().encode(TOKENIZER_CORPUS).length