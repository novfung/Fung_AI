from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer

model_name = 'smilegate-ai/kor_unsmile'
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=-1,   # CPU로 설정
        return_all_scores=True,
        function_to_apply='sigmoid'
    )
for result in pipe("안녕하세요!")[0]:
    if result['label'] == 'clean':
        clean_score = result['score']
        print(clean_score)