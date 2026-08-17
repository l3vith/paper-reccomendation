"""
Module for fine-tuning SciBERT using TripletLoss via sentence-transformers.
"""

import os
import torch
import logging
import argparse
import pandas as pd
from typing import Tuple, List, Any
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_training_data(triplets_path: str, test_split: float = 0.2) -> Tuple[List[InputExample], List[InputExample]]:
    """
    Load triplets and split into train/test datasets.
    
    Args:
        triplets_path (str): Path to the triplets CSV.
        test_split (float): Fraction of data to use for testing.
        
    Returns:
        Tuple[List[InputExample], List[InputExample]]: Train and test examples.
    """
    logging.info(f"Loading training data from {triplets_path}")
    df = pd.read_csv(triplets_path)
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=test_split, random_state=42)
    
    def df_to_examples(dataframe):
        examples = []
        for _, row in dataframe.iterrows():
            # Triplet examples contain texts and optional label
            examples.append(InputExample(texts=[row['anchor_text'], row['positive_text'], row['negative_text']]))
        return examples
        
    train_examples = df_to_examples(train_df)
    test_examples = df_to_examples(test_df)
    
    logging.info(f"Created {len(train_examples)} train examples and {len(test_examples)} test examples")
    return train_examples, test_examples

from sentence_transformers import SentenceTransformer, InputExample, losses, models

def create_model(
    model_name: str = 'allenai/scibert_scivocab_uncased',
    max_seq_length: int = 256,
    use_qlora: bool = True,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05
) -> SentenceTransformer:
    """
    Load and configure the SentenceTransformer model with optional Q-LoRA (4-bit NF4) adaptation.
    
    Args:
        model_name (str): HuggingFace model name.
        max_seq_length (int): Maximum sequence length.
        use_qlora (bool): Whether to apply 4-bit NormalFloat4 Quantization with LoRA (Q-LoRA).
        use_lora (bool): Whether to apply Low-Rank Adaptation (LoRA).
        lora_r (int): LoRA rank dimension.
        lora_alpha (int): LoRA scaling alpha.
        lora_dropout (float): Dropout probability for LoRA layers.
        
    Returns:
        SentenceTransformer: Configured model.
    """
    logging.info(f"Loading base scientific model: {model_name}")
    
    model_kwargs = {}
    if use_qlora:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            # Only pass quantization config if running on CUDA or supported platform
            if torch.cuda.is_available():
                model_kwargs["quantization_config"] = bnb_config
                logging.info("Enabled 4-bit NF4 Quantization (BitsAndBytes) for Q-LoRA.")
            else:
                logging.info("Hardware does not have native CUDA 4-bit kernels; running LoRA adaptation layer.")
        except Exception as e:
            logging.warning(f"Could not initialize 4-bit BitsAndBytesConfig: {e}. Falling back to standard LoRA.")

    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length, model_args=model_kwargs)
    
    if use_qlora or use_lora:
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            logger = logging.getLogger(__name__)
            
            # Prepare model for quantized training if 4-bit enabled
            if use_qlora and hasattr(word_embedding_model.auto_model, "is_loaded_in_4bit") and word_embedding_model.auto_model.is_loaded_in_4bit:
                word_embedding_model.auto_model = prepare_model_for_kbit_training(word_embedding_model.auto_model)
                logger.info("Model prepared for k-bit (4-bit) Q-LoRA training.")

            logger.info(f"Applying Q-LoRA / LoRA adapters (rank={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})...")
            
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["query", "value", "key", "dense"],
                lora_dropout=lora_dropout,
                bias="none"
            )
            word_embedding_model.auto_model = get_peft_model(word_embedding_model.auto_model, lora_config)
            
            # Log parameter count
            trainable_params = sum(p.numel() for p in word_embedding_model.auto_model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in word_embedding_model.auto_model.parameters())
            pct = 100 * trainable_params / all_params
            logger.info(f"Q-LoRA initialized: {trainable_params:,} trainable / {all_params:,} total params ({pct:.2f}%)")
        except ImportError:
            logging.warning("PEFT not found. Proceeding with full fine-tuning.")
        except Exception as e:
            logging.warning(f"Could not apply Q-LoRA: {e}. Falling back to full fine-tuning.")

    pooling_dim = word_embedding_model.get_word_embedding_dimension()
    pooling_model = models.Pooling(pooling_dim, pooling_mode="mean")
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

def train(model: SentenceTransformer, train_examples: List[InputExample], 
          output_path: str = 'models/scibert-finetuned-papers', 
          epochs: int = 2, batch_size: int = 8, 
          learning_rate: float = 2e-5, warmup_steps: int = 50) -> None:
    """
    Fine-tune the model using TripletLoss.
    
    Args:
        model (SentenceTransformer): The base model.
        train_examples (List[InputExample]): Training examples.
        output_path (str): Where to save the fine-tuned model.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        learning_rate (float): Learning rate.
        warmup_steps (int): Number of warmup steps for learning rate scheduler.
    """
    import os
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    logging.info("Preparing DataLoader and Loss function")
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False
    )
    
    # Use TripletLoss with COSINE distance and margin 0.5
    train_loss = losses.TripletLoss(model=model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.5)
    
    logging.info(f"Starting training for {epochs} epochs (batch_size={batch_size})")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': learning_rate},
        show_progress_bar=True
    )
    
    # If LoRA was used, merge adapter weights into a clean base model before saving
    try:
        if len(model) > 0 and hasattr(model[0], 'auto_model'):
            if hasattr(model[0].auto_model, 'merge_and_unload'):
                logging.info("Merging LoRA adapter weights into base SciBERT model...")
                merged_base = model[0].auto_model.merge_and_unload()
                
                # Re-wrap into clean standard SentenceTransformer
                clean_transformer = models.Transformer(
                    model_name_or_path='allenai/scibert_scivocab_uncased',
                    max_seq_length=256
                )
                clean_transformer.auto_model = merged_base
                clean_transformer.tokenizer = model[0].tokenizer
                pooling_model = model[1]
                
                clean_model = SentenceTransformer(modules=[clean_transformer, pooling_model])
                clean_model.save(output_path)
                logging.info(f"Training complete. Clean merged production model saved to {output_path}")
                return
    except Exception as e:
        logging.warning(f"Could not merge LoRA weights cleanly: {e}")

    model.save(output_path)
    logging.info(f"Training complete. Model saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune SciBERT model with Triplet Loss and Q-LoRA")
    parser.add_argument('--triplets', type=str, default='data/triplets.csv', help='Path to triplets CSV file')
    parser.add_argument('--output', type=str, default='models/scibert-finetuned-papers', help='Output path for saved model')
    parser.add_argument('--base-model', type=str, default='allenai/scibert_scivocab_uncased', help='Base model to fine-tune')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--qlora', action='store_true', default=True, help='Use 4-bit Q-LoRA parameter-efficient fine-tuning (default: True)')
    parser.add_argument('--no-qlora', action='store_false', dest='qlora', help='Disable Q-LoRA (perform full fine-tuning)')
    parser.add_argument('--lora-r', type=int, default=16, help='LoRA rank')
    
    args = parser.parse_args()
    
    print("\n--- Training Configuration ---")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("------------------------------\n")
    
    train_data, _ = load_training_data(args.triplets)
    model = create_model(args.base_model, use_qlora=args.qlora, lora_r=args.lora_r)
    train(model, train_data, output_path=args.output, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
