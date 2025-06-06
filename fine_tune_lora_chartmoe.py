from peft import get_peft_model, LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict, List, Optional, Sequence
from utils import process_data
from dataset import SciVQADataset
from tqdm import tqdm
import wandb
from torchvision import transforms
from torchvision.transforms import InterpolationMode


vis_processor = transforms.Compose([
            transforms.Resize((490, 490),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])
def collate_fn(batch):
    sample = {"data_type": ["multi" for _ in range(len(batch))]}
    images = [vis_processor(batch[i].image).unsqueeze(0)  for i in range(len(batch))]
    processed_data = [process_data(batch[idx].data) for idx in range(len(batch))]
    sample["text_input"] = [ [f'[UNUSED_TOKEN_146]user\n{"<ImageHere> "+ processed_data[i][0]}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n{processed_data[i][1]}[UNUSED_TOKEN_145]\n'] for i in range(len(batch))]
    sample['image'] = images

    return sample

def main(args):
    if not args.quick_debug:
        wandb.login()
        run = wandb.init(project="SciVQA", name="finetune_lora_chartmoe")
    
    model = AutoModelForCausalLM.from_pretrained("IDEA-FinAI/chartmoe",trust_remote_code=True).bfloat16().cuda()
    tokenizer = AutoTokenizer.from_pretrained("IDEA-FinAI/chartmoe",trust_remote_code=True)
    model.tokenizer = tokenizer
    for name, param in model.model.named_parameters():
            param.requires_grad = False
    model.vision_proj.requires_grad_(True) #No re-aligment required If set to True, need to train Auxillary Router with new embedding
    model.vit.requires_grad_(False)
    lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.lora_target_modules,
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
                task_type='CAUSAL_LM',
                modules_to_save=["vision_proj"],
            )
    model = get_peft_model(model, lora_config)
    if args.quick_debug:
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
    model.print_trainable_parameters()
    train_ds = SciVQADataset(args.train_data, args.train_image_root)
    val_ds = SciVQADataset(args.val_data, args.val_image_root)
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.num_epochs, T_mult=2, eta_min=1e-6)

    if args.quick_debug:
        args.num_epochs = 1
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for idx, batch in enumerate(tqdm(train_dataloader)):
            with torch.cuda.amp.autocast():
                outputs = model(samples=batch)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step(epoch + idx / len(train_dataloader))
                total_loss += loss.item()
                if not args.quick_debug:
                    run.log({"train_step_loss": loss.item()})
                if args.quick_debug and idx > 10:
                    break

        print(f"Epoch {epoch} total loss: {total_loss / len(train_dataloader)}")
        if not args.quick_debug:
            run.log({"train_epoch_loss": total_loss / len(train_dataloader)}, step=epoch)
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(val_dataloader)):
                with torch.cuda.amp.autocast():
                    outputs = model(samples=batch)
                    loss = outputs.loss
                    total_loss += loss.item()
                    if args.quick_debug and idx > 10:
                        break
        print(f"Epoch {epoch} validation loss: {total_loss / len(val_dataloader)}")
        if not args.quick_debug:
            run.log({"val_epoch_loss": total_loss / len(val_dataloader)}, step=epoch)
        model.save_pretrained(args.adapter_path)
        if args.quick_debug:
            del model
            model = AutoModelForCausalLM.from_pretrained("IDEA-FinAI/chartmoe",trust_remote_code=True).bfloat16().cuda()
            model =PeftModel.from_pretrained(model, args.adapter_path)
            model = model.merge_and_unload()
            model.save_pretrained("finetuned_chartmoe")
            del model
            model = AutoModelForCausalLM.from_pretrained("finetuned_chartmoe",trust_remote_code=True).bfloat16().cuda()
            print(model)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="/scratch/sc11537/SciVQA/train_cleaned.json")
    parser.add_argument("--val_data", type=str, default="/scratch/sc11537/SciVQA/validation_cleaned.json")
    parser.add_argument("--train_image_root", type=str, default="/scratch/sc11537/SciVQA/images_train")
    parser.add_argument("--val_image_root", type=str, default="/scratch/sc11537/SciVQA/images_validation")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_target_modules", type=List[str], default=[
        'attention.wqkv',
        'attention.wo',
        'feed_forward.w1',
        'feed_forward.w2',
        'feed_forward.w3',
    ])
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_bias", type=str, default="none")
    parser.add_argument("--adapter_path", type=str, default="/scratch/sc11537/SciVQA/finetune_lora_adapter")
    parser.add_argument("--quick_debug",action="store_true")
    args = parser.parse_args()
    main(args)