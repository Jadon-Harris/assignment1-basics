import argparse
import os

import numpy as np
import wandb
import torch
from tqdm import tqdm

from cs336_basics.train.adamw import AdamW
from cs336_basics.train.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.train.cross_entropy import cross_entropy
from cs336_basics.train.data_loading import get_batch
from cs336_basics.train.gradient_clippint import gradient_clipping
from cs336_basics.train.lr_cosine_schedule import lr_cosine_schedule
from cs336_basics.transformer.transformer_lm import TransformerLM


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer LLM model")
    # transformer_lm hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000, help="size of the vocabulary")
    parser.add_argument("--context_length", type=int, default=256, help="context length")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--d_ff", type=int, default=1344, help="fnn dimension of model")
    parser.add_argument("--num_heads", type=int, default=16, help="number of heads")
    parser.add_argument("--num_layers", type=int, default=4, help="number of layers")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="rope_theta")
    # Optimizer hyperparameters
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="adamW beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="adamW beta2")
    parser.add_argument("--eps", type=float, default=1e-8, help="adamW epsilon")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="gradient clipping norm")
    parser.add_argument("--max_lr", type=float, default=1e-3, help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-4, help="Minimum learning rate")
    parser.add_argument("--warm_up_it", type=int, default=500, help="Warmup iterations")
    parser.add_argument("--cosine_it", type=int, default=5000, help="Cosine annealing iterations")
    # train hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--train_steps", type=int, default=5000, help="number of training steps")
    parser.add_argument("--val_interval", type=int, default=100, help="validation interval")
    parser.add_argument("--val_batch_size", type=int, default=10, help="validation batch size")
    parser.add_argument("--save_interval", type=int, default=1000, help="checkpoint save interval")
    parser.add_argument("--log_interval", type=int, default=1, help="logging interval")
    parser.add_argument("--save_ckpt_path", type=str, default="../../checkpoints", help="checkpoint save directory")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="path to the checkpoint resume from")
    # Data hyperparameters
    parser.add_argument("--data_dir", type=str, default="../../data", help="data directory")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, mps")
    # Wandb
    parser.add_argument("--wandb_project", type=str, default="cs336-transformer", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")

    return parser.parse_args()


def get_device(device_arg):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_arg


def get_dataset_memmap(path, dtype=np.uint32):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    dataset = np.memmap(path, dtype=dtype, mode="r")
    return dataset


def main():
    args = parse_args()

    # Show device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Init wandb
    wandb.init(project=args.wandb_project,
               name=args.wandb_run_name,
               config=vars(args))
    print(f"wandb initialized: {args.wandb_run_name}")

    # create checkpoint dir
    os.makedirs(args.save_ckpt_path, exist_ok=True)

    # Init Model
    model = TransformerLM(vocab_size=args.vocab_size,
                          context_length=args.context_length,
                          d_model=args.d_model,
                          d_ff=args.d_ff,
                          num_heads=args.num_heads,
                          num_layers=args.num_layers,
                          rope_theta=args.rope_theta
                          ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    wandb.log({"Total number of parameters": total_params})

    # Init optimizer
    optimizer = AdamW(model.parameters(),
                      lr=args.max_lr,
                      betas=(args.beta1, args.beta2),
                      eps=args.eps,
                      weight_decay=args.weight_decay)

    train_data_path = os.path.join(args.data_dir, "train.dat")
    val_data_path = os.path.join(args.data_dir, "valid.dat")

    train_data = get_dataset_memmap(train_data_path)
    val_data = get_dataset_memmap(val_data_path)

    print(f"train_data size: {len(train_data)} tokens")
    print(f"val_data shape: {len(val_data)} tokens")

    # Resum checkpoint
    start_iter = 0
    if args.resume_ckpt is not None:
        print(f"loading checkpoint from {args.resume_ckpt}")
        start_iter = load_checkpoint(args.resume_ckpt, model, optimizer)
        print(f"resum from iteration {start_iter}")

    # train loop
    model.train()
    train_loss = []

    # init progress bar
    pbar = tqdm(range(start_iter, args.train_steps),
                desc="Training",
                initial=start_iter,
                total=args.train_steps)

    for iter in pbar:
        # get cur lr and update
        lr = lr_cosine_schedule(iter, args.max_lr, args.min_lr, args.warm_up_it, args.cosine_it)

        for group in optimizer.param_groups:
            group['alpha'] = lr

        # sample batch
        input_ids, target_ids = get_batch(train_data, batch_size=args.batch_size, context_length=args.context_length,
                                          device=device)
        input_ids = input_ids.long().to(device)
        target_ids = target_ids.long().to(device)

        # forward
        optimizer.zero_grad()
        logits = model(input_ids)

        # cal loss
        logits_flat = logits.view(-1, logits.size(-1))  # reshape to (batch_size * seq_len, vocab_size)
        target_flat = target_ids.view(-1)  # reshape to (batch_size * seq_len)

        loss = cross_entropy(logits_flat, target_flat)

        # backward
        loss.backward()

        # gradient clipping
        gradient_clipping(model.parameters(), args.clip_grad_norm)

        # Optimizer step
        optimizer.step()

        # track loss
        train_loss.append(loss.item())

        # log
        if iter % args.log_interval == 0:
            avg_loss = np.mean(train_loss[-100:]) if len(train_loss) >= 100 else np.mean(train_loss)
            perplexity = np.exp(avg_loss)

            pbar.set_postfix({"loss": f'{loss.item():.4f}',
                              "avg_loss": f'{avg_loss:.4f}',
                              "perplexity": f'{perplexity:.2f}',
                              "lr": f'{lr:.2e}'})

            wandb.log({"train/loss": loss.item(),
                       "train/avg_loss": avg_loss,
                       "train/perplexity": perplexity,
                       "train/lr": lr,
                       "iteration": iter
                       })

        # validation
        if iter % args.val_interval == 0 and iter > 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(args.val_batch_size):
                    val_input_ids, val_target_ids = get_batch(val_data, batch_size=args.batch_size,
                                                              context_length=args.context_length, device=device)
                    val_input_ids = val_input_ids.long().to(device)
                    val_target_ids = val_target_ids.long().to(device)

                    val_logits = model(val_input_ids)
                    val_logits_flat = val_logits.view(-1, val_logits.size(-1))
                    val_target_flat = val_target_ids.view(-1)

                    val_loss = cross_entropy(val_logits_flat, val_target_flat)
                    val_losses.append(val_loss.item())
            avg_val_loss = np.mean(val_losses)
            val_perplexity = np.exp(avg_val_loss)

            tqdm.write(f"Validation | loss: {avg_val_loss:.4f} | perplexity: {val_perplexity:.2f}")
            wandb.log({"val/loss": avg_val_loss,
                       "val/perplexity": val_perplexity,
                       "iteration": iter})

            model.train()

        if iter % args.save_interval == 0 and iter > 0:
            checkpoint_path = os.path.join(args.save_ckpt_path, f"checkpoint_{iter}.pt")
            save_checkpoint(model, optimizer, iter, checkpoint_path)
            tqdm.write(f"saving checkpoint: {checkpoint_path}")

    pbar.close()

    final_checkpoint = os.path.join(args.save_ckpt_path, f"final_checkpoint_{args.train_steps}.pt")
    save_checkpoint(model, optimizer, args.train_steps, final_checkpoint)
    print(f"saving final checkpoint: {final_checkpoint}")
    print('training finished!')

    wandb.finish()


if __name__ == '__main__':
    main()
