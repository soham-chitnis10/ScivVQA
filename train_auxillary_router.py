from transformers import AutoImageProcessor, AutoTokenizer, AutoModel
import argparse
from utils import process_data
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import SciVQADataset
from tqdm import tqdm
import torch
from models import AuxillaryRouter
from sklearn.metrics import f1_score, accuracy_score
from torch import nn
import numpy as np

def label_data(data):
    if data['qa_pair_type'] == "unanswerable":
        return 1
    else:
        return 0
    
def collate_fn(batch):
    images = [batch[i].image for i in range(len(batch))]
    texts = [process_data(batch[i].data)[0] for i in range(len(batch))]
    labels = [label_data(batch[i].data) for i in range(len(batch))]
    return images, texts, labels

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def focal_loss(pred, targets, alpha=0.25, gamma=2.0):
    log_prob = torch.log_softmax(pred,dim=1)
    one_hot_target = nn.functional.one_hot(targets,log_prob.shape[1])
    weight = alpha*torch.pow(1- log_prob.exp(),gamma)
    loss = torch.mean((-1*weight*log_prob)*one_hot_target)
    return loss

def main(args):
    train_ds = SciVQADataset(args.train_json_file, args.train_image_root)
    class_count = np.array([12960, 2160])
    weight  = 1.0 / class_count
    sample_weight = []
    for i in range(len(train_ds)):
        if train_ds.df['qa_pair_type'][i] == "unanswerable":
            sample_weight.append(weight[1])
        else:
            sample_weight.append(weight[0])
    sample_weight = torch.tensor(sample_weight, dtype=torch.float)
    sampler = WeightedRandomSampler(sample_weight, len(sample_weight))
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, sampler = sampler, num_workers=args.num_workers, collate_fn=collate_fn)
    val_ds = SciVQADataset(args.val_json_file, args.val_image_root)
    val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    chartmoe = AutoModel.from_pretrained('finetuned_chartmoe_new',trust_remote_code=True).eval().cuda()
    chartmoe = freeze_model(chartmoe)
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    text_encoder = AutoModel.from_pretrained("google-bert/bert-base-uncased").eval().cuda()
    text_encoder = freeze_model(text_encoder)
    aux_router = AuxillaryRouter(visual_embed=4096,text_embed_dim=768).cuda()
    loss_fn = focal_loss if args.use_focal else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(aux_router.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.num_epochs, T_mult=2, eta_min=1e-6)
    best_f1 = 0

    for epoch in range(args.num_epochs):
        total_loss = 0
        predictions = []
        all_labels = []
        for idx, batch in enumerate(tqdm(train_dataloader)):
            images, texts, labels = batch
            images = torch.cat([chartmoe.vis_processor(image).cuda().unsqueeze(0) for image in images], dim=0)
            textual_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            labels = torch.tensor(labels).cuda()

            # Process images and texts
            image_inputs = chartmoe.encode_img(images)
            text_inputs = text_encoder(**textual_inputs)

            
            logits = aux_router(image_inputs, text_inputs.last_hidden_state)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + idx / len(train_dataloader))
            total_loss += loss.item()
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Forward pass
        acc = accuracy_score(all_labels, predictions)
        f1 = f1_score(all_labels, predictions)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {total_loss/len(train_dataloader)}, Train Accuracy: {acc}, Train F1 Score: {f1}")

        total_loss = 0
        predictions = []
        all_labels = []
        for idx, batch in tqdm(enumerate(val_dataloader)):
            with torch.no_grad():
                images, texts, labels = batch
                images = torch.cat([chartmoe.vis_processor(image).cuda().unsqueeze(0) for image in images], dim=0)
                textual_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
                labels = torch.tensor(labels).cuda()
                

                # Process images and texts
                image_inputs = chartmoe.encode_img(images)
                text_inputs = text_encoder(**textual_inputs)
                logits = aux_router(image_inputs, text_inputs.last_hidden_state)
                loss = loss_fn(logits, labels)
                total_loss += loss.item()
                predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, predictions)
        f1 = f1_score(all_labels, predictions)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Val Loss: {total_loss/len(val_dataloader)}, Val Accuracy: {acc}, Val F1 Score: {f1}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(aux_router.state_dict(), args.model_path)
            print(f"Model saved at epoch {epoch+1} with F1 score: {best_f1}")
    print("Training complete. Best F1 score: ", best_f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json_file", type=str, required=True, help="Path to train json file")
    parser.add_argument("--train_image_root", type=str, required=True, help="Path to train image root")
    parser.add_argument("--val_json_file", type=str, required=True, help="Path to val json file")
    parser.add_argument("--val_image_root", type=str, required=True, help="Path to val image root")
    parser.add_argument("--lr", type=float, default=1e-4, help="Path to the predictions")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--use_focal", default=False, action="store_true")
    args = parser.parse_args()
    main(args)