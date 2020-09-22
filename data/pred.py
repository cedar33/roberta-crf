from fairseq.models.roberta import RobertaModel
label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)

roberta = RobertaModel.from_pretrained(
    model_name_or_path='/path/to/checkpoints',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='/path/to/data'
)
roberta.eval() 
tokens = roberta.encode('china is a great country')
pred = roberta.predict_label("kaggle_ner", tokens, return_logits=True)

print(pred[1][0])
print([label_fn(int(p)) for p in pred[1][0].tolist()])
