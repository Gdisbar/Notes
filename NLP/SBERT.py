from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F
tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
model = AutoModel.from_pretrained('deepset/sentence_bert')

sentence = 'Who are you voting for in 2020?'
labels = ['business', 'art & culture', 'politics']

# run inputs through model and mean-pool over the sequence
# dimension to get sequence-level representations
inputs = tokenizer.batch_encode_plus([sentence] + labels,
                                     return_tensors='pt',
                                     pad_to_max_length=True)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
output = model(input_ids, attention_mask=attention_mask)[0]
sentence_rep = output[:1].mean(dim=1)
label_reps = output[1:].mean(dim=1)

# now find the labels with the highest cosine similarities to
# the sentence
similarities = F.cosine_similarity(sentence_rep, label_reps)
closest = similarities.argsort(descending=True)
for ind in closest:
    print(f'label: {labels[ind]} \t similarity: {similarities[ind]}')

# One problem with this method is that Sentence-BERT is designed to learn effective sentence-level, 
# not single- or multi-word representations like our class names. It is therefore reasonable to suppose 
# that our label embeddings may not be as semantically salient as popular word-level embedding methods (i.e. word2vec). 
# If we were to use word vectors as our label representations, however, we would need annotated data to learn 
# an alignment between the S-BERT sequence representations and the word2vec label representations.    

#---------------------------------------------------------------------------------------------------------------------------------

# When using transformer architectures like BERT, NLI datasets are typically modeled 
# via sequence-pair classification. That is, we feed both the premise and the hypothesis 
# through the model together as distinct segments and learn a classification head predicting 
# one of [contradiction, neutral, entailment].

# load model pretrained on MNLI
from transformers import BartForSequenceClassification, BartTokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli')
model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')

# pose sequence as a NLI premise and label (politics) as a hypothesis
premise = 'Who are you voting for in 2020?'
hypothesis = 'This text is about politics.'

# run through model pre-trained on MNLI
input_ids = tokenizer.encode(premise, hypothesis, return_tensors='pt')
logits = model(input_ids)[0]

# we throw away "neutral" (dim 1) and take the probability of
# "entailment" (2) as the probability of the label being true 
entail_contradiction_logits = logits[:,[0,2]]
probs = entail_contradiction_logits.softmax(dim=1)
true_prob = probs[:,1].item() * 100
print(f'Probability that the label is true: {true_prob:0.2f}%')

## -----------------------------------------------------------------------------------------------------------------------------------
# SetFit - Efficient Few-shot Learning with Sentence Transformers -- No prompt required

from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss

from setfit import SetFitModel, SetFitTrainer, sample_dataset


# Load a dataset from the Hugging Face Hub
dataset = load_dataset("sst2")

# Simulate the few-shot regime by sampling 8 examples per class
train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=8)
eval_dataset = dataset["validation"]

# Load a SetFit model from Hub
model = SetFitModel.from_pretrained(
    "sentence-transformers/paraphrase-mpnet-base-v2",
    use_differentiable_head=True,
    head_params={"out_features": num_classes},
)



# model = SetFitModel.from_pretrained(
#     model_id,
#     multi_target_strategy="one-vs-rest",   ## 
# )
# one-vs-rest: uses a OneVsRestClassifier head.
# multi-output: uses a MultiOutputClassifier head.
# classifier-chain: uses a ClassifierChain head.
# model = SetFitModel.from_pretrained(
#     model_id,
#     multi_target_strategy="one-vs-rest"
#     use_differentiable_head=True,
#     head_params={"out_features": num_classes},
# )
# If you use the differentiable SetFitHead classifier head, it will automatically use 
# BCEWithLogitsLoss for training. The prediction involves a sigmoid after which probabilities 
# are rounded to 1 or 0. Furthermore, the "one-vs-rest" and "multi-output" multi-target strategies are 
# equivalent for the differentiable SetFitHead.


# Create trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss_class=CosineSimilarityLoss,
    metric="accuracy",
    batch_size=16,
    num_iterations=20, # The number of text pairs to generate for contrastive learning
    num_epochs=1, # The number of epochs to use for contrastive learning
    column_mapping={"sentence": "text", "label": "label"} # Map dataset columns to text/label expected by trainer
)

# Train and evaluate
trainer.freeze() # Freeze the head
trainer.train() # Train only the body

# Unfreeze the head and freeze the body -> head-only training
trainer.unfreeze(keep_body_frozen=True)
# or
# Unfreeze the head and unfreeze the body -> end-to-end training
trainer.unfreeze(keep_body_frozen=False)

trainer.train(
    num_epochs=25, # The number of epochs to train the head or the whole model (body and head)
    batch_size=16,
    body_learning_rate=1e-5, # The body's learning rate
    learning_rate=1e-2, # The head's learning rate
    l2_weight=0.0, # Weight decay on **both** the body and head. If `None`, will use 0.01.
)
metrics = trainer.evaluate()

# Push model to the Hub
trainer.push_to_hub("my-awesome-setfit-model")

# Download from Hub and run inference
model = SetFitModel.from_pretrained("lewtun/my-awesome-setfit-model")
# Run inference
preds = model(["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"])

## hyper-parameter tuning

def model_init(params):
    params = params or {}
    max_iter = params.get("max_iter", 100)
    solver = params.get("solver", "liblinear")
    params = {
        "head_params": {
            "max_iter": max_iter,
            "solver": solver,
        }
    }
    return SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2", **params)


def hp_space(trial):  # Training parameters
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_epochs": trial.suggest_int("num_epochs", 1, 5),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64]),
        "seed": trial.suggest_int("seed", 1, 40),
        "num_iterations": trial.suggest_categorical("num_iterations", [5, 10, 20]),
        "max_iter": trial.suggest_int("max_iter", 50, 300),
        "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"]),
    }


dataset = Dataset.from_dict(
            {"text_new": ["a", "b", "c"], "label_new": [0, 1, 2], "extra_column": ["d", "e", "f"]}
        )

trainer = SetFitTrainer(
    train_dataset=dataset,
    eval_dataset=dataset,
    model_init=model_init,
    column_mapping={"text_new": "text", "label_new": "label"},
)
best_run = trainer.hyperparameter_search(direction="maximize", hp_space=hp_space, n_trials=20)
trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)
trainer.train()

## Compressing a SetFit model with knowledge distillation

from sentence_transformers.losses import CosineSimilarityLoss

from setfit import SetFitModel, SetFitTrainer, DistillationSetFitTrainer, sample_dataset

# Load a dataset from the Hugging Face Hub
dataset = load_dataset("ag_news")

# Create a sample few-shot dataset to train the teacher model
train_dataset_teacher = sample_dataset(dataset["train"], label_column="label", num_samples=16)
# Create a dataset of unlabeled examples to train the student
train_dataset_student = dataset["train"].shuffle(seed=0).select(range(500))
# Dataset for evaluation
eval_dataset = dataset["test"]

# Load teacher model
teacher_model = SetFitModel.from_pretrained(
    "sentence-transformers/paraphrase-mpnet-base-v2"
)

# Create trainer for teacher model
teacher_trainer = SetFitTrainer(
    model=teacher_model,
    train_dataset=train_dataset_teacher,
    eval_dataset=eval_dataset,
    loss_class=CosineSimilarityLoss,
)

# Train teacher model
teacher_trainer.train()

# Load small student model
student_model = SetFitModel.from_pretrained("paraphrase-MiniLM-L3-v2")

# Create trainer for knowledge distillation
student_trainer = DistillationSetFitTrainer(
    teacher_model=teacher_model,
    train_dataset=train_dataset_student,
    student_model=student_model,
    eval_dataset=eval_dataset,
    loss_class=CosineSimilarityLoss,
    metric="accuracy",
    batch_size=16,
    num_iterations=20,
    num_epochs=1,
)

# Train student with knowledge distillation
student_trainer.train()