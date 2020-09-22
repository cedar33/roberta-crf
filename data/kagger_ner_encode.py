from fairseq.data.encoders.gpt2_bpe import get_encoder
encoder = get_encoder("/path/to/roberta.base/encoder.json", "/path/to/roberta.base/vocab.bpe")
train_data = []
with open("dev.csv", "r", encoding="utf-8") as f:
    lines = f.readlines()
    tmp_sentence = []
    for idx, line in enumerate(lines):
        if idx == 0:
            continue
        if line.split(",")[0].startswith("Sentence: "):
            if len(tmp_sentence) > 0:
                train_data.append(tmp_sentence)
            tmp_sentence = []
        line = line.strip().split(",")
        if len(line) == 4:
            word = line[1]
            label = line[3]
        else:
            word = ","
            label = line[-1]
        ids = encoder.encode(word)
        for idx, _id in enumerate(ids):
            if label.startswith("B") and idx != 0:
                label = "I"+label[1:]
            tmp_sentence.append((str(_id), label))

with open("dev.text.txt.bpe", "w", encoding="utf-8") as ft, \
    open("dev.label.txt", "w", encoding="utf-8") as fl:
    for idx, t in enumerate(train_data):
        if idx % 10000 == 0:
            print(idx)
        text = [_t[0] for _t in t]
        label = [_t[1] for _t in t]
        assert len(text) == len(label)
        ft.write(" ".join(text)+"\n")
        fl.write(" ".join(label)+"\n")
