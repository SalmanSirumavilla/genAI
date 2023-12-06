import argparse

def from_torch_onnx():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # load model and tokenizer
    model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dummy_model_input = tokenizer("This is a sample", return_tensors="pt")

    # export
    torch.onnx.export(
        model, 
        tuple(dummy_model_input.values()),
        f="torch-model.onnx",  
        input_names=['input_ids', 'attention_mask'], 
        output_names=['logits'], 
        dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}, 
                    'attention_mask': {0: 'batch_size', 1: 'sequence'}, 
                    'logits': {0: 'batch_size', 1: 'sequence'}}, 
        do_constant_folding=True, 
        opset_version=13, 
    )


def from_transformers_onnx():
    from pathlib import Path
    import transformers
    from transformers.onnx import FeaturesManager
    from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

    # load model and tokenizer
    model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    feature = "sequence-classification"
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # load config
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
    onnx_config = model_onnx_config(model.config)

    # export
    onnx_inputs, onnx_outputs = transformers.onnx.export(
            preprocessor=tokenizer,
            model=model,
            config=onnx_config,
            opset=13,
            output=Path("trfs-model.onnx")
    )

def from_optimum_onnx():
    from optimum.onnxruntime import ORTModelForSequenceClassification

    model = ORTModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english",from_transformers=True)

def main():
    parser = argparse.ArgumentParser(description="Script to call different onnx conversion functions.")
    parser.add_argument("--onnx_func", type=str, choices=["torch", "transformer", "optimum"], help="Choose a function to execute (torch, transformer, or optimum)")
    args = parser.parse_args()

    if args.onnx_func == "torch":
        from_torch_onnx()
    elif args.onnx_func == "transformer":
        from_transformers_onnx()
    elif args.onnx_func == "optimum":
        from_optimum_onnx()

if __name__ == "__main__":
    main()