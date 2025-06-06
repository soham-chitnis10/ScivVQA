from dataset import SciVQADataset
from torch.utils.data import DataLoader
from chartmoe import ChartMoE_Robot
from tqdm import tqdm
import torch
import pandas as pd
import argparse
from models import AuxillaryRouter
from transformers import AutoModel, AutoTokenizer
from utils import process_data


def infer_aux_router(aux_router, chartmoe, text_encoder, image, textual_inputs):
    with torch.no_grad():
        image = chartmoe.model.vis_processor(image).cuda().unsqueeze(0)
        image_inputs = chartmoe.model.encode_img(image)
        text_inputs = text_encoder(**textual_inputs)
        logits = aux_router(image_inputs, text_inputs.last_hidden_state)
        return logits

def main(args):
    ds = SciVQADataset(args.json_file,args.image_root)
    df = pd.DataFrame(columns=['instance_id','answer_pred'])

    robot = ChartMoE_Robot(ckpt_path="finetuned_chartmoe_new")
    aux_router = AuxillaryRouter(visual_embed=4096,text_embed_dim=768)
    aux_router.load_state_dict(torch.load("/scratch/sc11537/SciVQA/aux_router_new_sampler_vision_proj_trained.pth"))
    aux_router.eval()
    aux_router = aux_router.cuda()
    text_encoder = AutoModel.from_pretrained("google-bert/bert-base-uncased").eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    for idx, batch in enumerate(tqdm(ds)):
        with torch.no_grad():
            image = batch.image
            data = batch.data
            question = process_data(data)
            if 'answer' in data.keys():
                question = question[0]
            with torch.cuda.amp.autocast():
                textual_inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
                logits = infer_aux_router(aux_router, robot, text_encoder, image, textual_inputs)
                label = torch.argmax(logits, dim=1).item()
                if label == 1:
                    response = "It is not possible to answer this question based only on the provided data."
                else:
                    response, _ = robot.chat(image=image, question=question)
                df.loc[len(df)]=[data['instance_id'],response]

    print("Saving predictions")
    df.to_csv(args.output_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, required=True, help="Path to json file")
    parser.add_argument("--image_root", type=str, required=True, help="Path to image root")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the predictions")
    args = parser.parse_args()
    main(args)