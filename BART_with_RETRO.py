import torch
from retro_pytorch import RETRO, TrainingWrapper
from transformers import BartTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import BertTokenizer

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

def tokenize_function(examples):
    return tokenizer(examples["text"])

def group_texts(examples):
    # Concatenate all texts.
    block_size=128
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
def train_bart_with_retro():
        retro = RETRO(
            chunk_size=128,
            max_seq_len = 2048,                      # max sequence length
            enc_dim = 896,                           # encoder model dimension
            enc_depth = 3,                           # encoder depth
            dec_dim = 768,                           # decoder model dimensions
            dec_depth = 12,                          # decoder depth
            dec_cross_attn_layers = (1, 3, 6, 9),    # decoder cross attention layers (with causal chunk cross attention)
            heads = 8,                               # attention heads
            dim_head = 64,                           # dimension per head
            dec_attn_dropout = 0.25,                 # decoder attention dropout
            dec_ff_dropout = 0.25                    # decoder feedforward dropout
        ).cuda()

        wrapper = TrainingWrapper(
          retro = retro,                                 # path to retro instance
          knn = 2,                                       # knn (2 in paper was sufficient)
          chunk_size = 128,                               # chunk size (64 in paper)
          documents_path = './text_folder',              # path to folder of text
          glob = '**/*.txt',                             # text glob
          chunks_memmap_path = './train.chunks.dat',     # path to chunks
          seqs_memmap_path = './train.seq.dat',          # path to sequence data
          doc_ids_memmap_path = './train.doc_ids.dat',   # path to document ids per chunk (used for filtering neighbors belonging to same document)
          max_chunks = 1_000_000,                        # maximum cap to chunks
          max_seqs = 100_000,                            # maximum seqs
          knn_extra_neighbors = 100,                     # num extra neighbors to fetch
          max_index_memory_usage = '100m',
          current_memory_available = '1G'
        )

        # get the dataloader
        train_dl = iter(wrapper.get_dataloader(batch_size = 2, shuffle = True))

        seq, retrieved = map(lambda t: t.cuda(), next(train_dl))
        #input to BART
        tokenizer_bert = BertTokenizer.from_pretrained('bert-base-cased')
        inputs = torch.cat([seq, retrieved], -1)
        b_i=[]
        for i in inputs:
            x = tokenizer_bert.convert_ids_to_tokens(i)
            b_i.append(x)
        
        b_i = pd.DataFrame({'text': b_i})
        dataset = Dataset.from_pandas(bart_inputs)




        # Load tokenizer and model
        #tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        model = AutoModelForCausalLM.from_pretrained('facebook/bart-base')
        #Adding training argument
        training_args = TrainingArguments(output_dir='./outputs',evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01, )
        

        # Preprocess dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
        lm_dataset = tokenized_dataset.map(group_texts, batched=True, batch_size=1000, num_proc=4)

        # Configure trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_dataset['train'],
            eval_dataset=lm_dataset['test']            
        )

        # Train
        train_result = trainer.train()
if __name__ == '__main__':
  train_bart_with_retro()