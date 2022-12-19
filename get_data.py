from datasets import load_dataset

datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
data=datasets['train']['text']
with open('text_folder/extracted_data.txt', 'w') as f:
    for line in data:
        f.write(line)
        f.write('\n')