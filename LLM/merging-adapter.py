# Combine Multiple LoRA adapters


hinglish_peft_model_id = "smangrul/mistral_hinglish_instruct_poc"
mental_health_peft_model_id = "GRMenon/mental-health-mistral-7b-instructv0.2-finetuned-V2"

from peft import PeftConfig,PeftModel 

config = PeftConfig.from_pretrained(hinglish_peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_id_or_path,load_in_4bit=True,device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(hinglish_peft_model_id)
model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(model,hinglish_peft_model_id,adapter_name="hinglish")
_ = model.load_adapter(mental_health_peft_model_id,adapter_name="mental-health")


adapter_list = ["hinglish","mental-health"]
weights = [07,0.3]
density = 0.7
combination_type="magnitude_prune"
# "linear_combination","concatenation","feature_fusion","attention_mechanism","gating_mechanism"
adapter_name = "hinglish-mental-health"
model.add_weighted_adapter(adapter_list,weights,
    density,combination_type,adapter_name)
model.set_adapter("hinglish-mental-health")
model.eval()

message = [{"role":"user","message":""}]
text = tokenizer.apply_chat_template(message,add_generation_prompt=True,tokenize=False)
inputs = tokenizer(text,return_tensors="pt")
inputs = {k : v.to("cuda") for k,v in inputs.items()}
outputs = model.generate(**inputs,max_new_tokens=100,
    do_sample=True,
    top_p=0.95,
    temperature=0.2,
    repetition_penalty=1.2,
    eos_token_id=tokenizer.eos_token_id,
)
result = tokenizer.decode(outputs[0])



from peft import Adapters

adapter_config_path = "path_to_your_adapter_config.json"
adapter = Adapters.load(adapter_config_path)
model.add_adapter(adapter)
model.load_adapter_weights(adapter_weights_path)



import torch
from torch import nn

class CustomFeatureFusionModel(nn.Module):
    def __init__(self, base_model, adapters, fusion_type="concatenation"):
        super(CustomFeatureFusionModel, self).__init__()
        self.base_model = base_model
        self.adapters = adapters
        self.fusion_type = fusion_type
        # gate : (hidden_size,no_of_adapters)
        self.gate = nn.Linear(base_model.config.hidden_size, len(adapters))  # Simple gating mechanism
        # concat(fusion_layer) : (no_of_adapters*hidden_size,hidden_size)
        # feature_fusion(fusion_layer) : (hidden_size,hidden_size)
        if fusion_type == "concatenation":
            self.fusion_layer = nn.Linear(len(adapters) * base_model.config.hidden_size, base_model.config.hidden_size)
        elif fusion_type == "feature_fusion":
            self.fusion_layer = nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size)
        else:
            raise ValueError(f"Fusion type {fusion_type} is not supported")

    def forward(self, input_ids, attention_mask=None):
        adapter_outputs = [adapter(input_ids, attention_mask=attention_mask) for adapter in self.adapters]
        
        if self.fusion_type == "concatenation":
            fused_output = torch.cat(adapter_outputs, dim=-1)
            fused_output = self.fusion_layer(fused_output)
        elif self.fusion_type == "feature_fusion":
            fused_output = torch.mean(torch.stack(adapter_outputs), dim=0)
            fused_output = self.fusion_layer(fused_output)
        
        # Apply gating
        gate_scores = torch.softmax(self.gate(fused_output), dim=-1)
        gated_output = sum(g * output for g, output in zip(gate_scores, adapter_outputs))
        
        return self.base_model(inputs_embeds=gated_output)

# Load base model and adapters
base_model_name = config.base_model_id_or_path
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, load_in_4bit=True, device_map="auto")
adapters = [PeftModel.from_pretrained(base_model, hinglish_peft_model_id, adapter_name="hinglish"),
             PeftModel.from_pretrained(base_model, mental_health_peft_model_id, adapter_name="mental-health")]

# Initialize custom model with feature fusion and gating mechanism
model = CustomFeatureFusionModel(base_model, adapters, fusion_type="feature_fusion")

# Move model to device
model = model.to("cuda")


from transformers import Trainer, TrainingArguments

# Define your training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('path_to_save_finetuned_model')
tokenizer.save_pretrained('path_to_save_finetuned_model')

"""
After training is complete, this saves the trained 
model to the directory "sft_santacoder1b".
"""
trainer.train()
trainer.save_model("sft_santacoder1b")
"""
saves the model’s state (weights and configuration) 
to this new directory. This step creates a snapshot of 
the model after training, which can be loaded later for 
evaluation or further use.
"""
output_dir = os.path.join("sft_santacoder1b", "final_checkpoint")
trainer.model.save_pretrained(output_dir)

# Free memory for merging weights
del base_model
torch.cuda.empty_cache()
"""
merges any components of the model (if it’s a model with 
adapters or other modular components) into a single model 
and unloads any unnecessary parts, optimizing the model 
for deployment or further use.

"""
model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.float16)
model = model.merge_and_unload()

output_merged_dir = os.path.join("sft_santacoder1b", "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)

# Fine-tune your model with adapters

# Save the base model
base_model.save_pretrained("sft_santacoder1b/base_model")

# Save adapters
model.save_adapter("sft_santacoder1b/adapters", adapter_name="my_adapter")

# Load the base model
model = AutoModelForCausalLM.from_pretrained("sft_santacoder1b/base_model", device_map="auto", torch_dtype=torch.float16)

# Load the adapters
model.load_adapter("sft_santacoder1b/adapters", adapter_name="my_adapter")

# Merge and unload the adapters
model = model.merge_and_unload()

# Save the final merged model
model.save_pretrained("sft_santacoder1b/final_merged_checkpoint", safe_serialization=True)



model = AutoModelForCausalLM.from_pretrained(
        "/content/sft_santacoder1b/final_merged_checkpoint/", 
        return_dict=True, torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(model, "/content/dpo_santacoder1b/final_checkpoint/")
model.eval()
model = model.merge_and_unload()
