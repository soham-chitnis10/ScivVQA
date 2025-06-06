# from transformers import AutoImageProcessor, AutoTokenizer, AutoModelForCausalLM, ConvNextV2Model
from torch import nn
import torch
from utils import process_data

# class ChartNeXtLlama(nn.Module):
#     def __init__(self):
#         super(ChartNeXtLlama, self).__init__()
#         self.image_processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-large-22k-384", use_fast=True)
#         self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
#         self.llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
#         self.convnextv2 = ConvNextV2Model.from_pretrained("facebook/convnextv2-large-22k-384")
#         self.connector = nn.Conv2d(1536,3072,1)
        
#     def forward(self, image_inputs, textual_inputs):
#         # Process image inputs
#         image_outputs = self.convnextv2(**image_inputs)
#         image_embeddings = image_outputs.last_hidden_state
#         image_tokens = self.connector(image_embeddings).permute(0,2,3,1).reshape(image_embeddings.shape[0], -1, 3072)
        
#         text_embedding = self.llm.get_input_embeddings()(textual_inputs["input_ids"])
#         embeddings = torch.cat([text_embedding, image_tokens], dim=1)
#         # Pass through LLM
#         textual_inputs['inputs_embeds'] = embeddings
#         textual_inputs['attention_mask'] = torch.cat([textual_inputs['attention_mask'], torch.ones((image_tokens.shape[0],image_tokens.shape[1]),device=embeddings.device)], dim=1)
#         textual_inputs.pop('input_ids',None)
#         outputs = self.llm(**textual_inputs)
        
#         return outputs


class AuxillaryRouter(nn.Module):
    def __init__(self, visual_embed, text_embed_dim,):
        super(AuxillaryRouter, self).__init__()
        if visual_embed != text_embed_dim:
            self.connector = nn.Linear(visual_embed, text_embed_dim)
        else:
            self.connector = None
        self.visual_embed = visual_embed
        self.text_embed_dim = text_embed_dim
        self.attention = nn.TransformerEncoder(nn.TransformerEncoderLayer(text_embed_dim, nhead=8, batch_first=True, activation='gelu', norm_first=True), num_layers=2, enable_nested_tensor=False)
        self.mlp = nn.Linear(text_embed_dim, 2)
    
    def forward(self, visual_embeddings, textual_embeddings):
        # Cross attention between visual and textual embeddings
        
        if self.connector is not None:
            visual_embeddings = self.connector(visual_embeddings)
        embeddings = torch.cat([torch.zeros((textual_embeddings.shape[0],1,textual_embeddings.shape[2]),device=textual_embeddings.device), visual_embeddings, textual_embeddings], dim=1)

        attn_output = self.attention(embeddings)
        # MLP to get logits
        logits = self.mlp(attn_output[:,0])

        return logits  
    
if __name__=="__main__":
    import dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = dataset.SciVQADataset("/scratch/sc11537/SciVQA/train_2025-03-27_18-34-44.json","/scratch/sc11537/SciVQA/images_train")
    vlm = ChartNeXtLlama().to(device)
    print(vlm)
    image_inputs = vlm.image_processor(images=[ds[0].image],size={"shortest_edge":1080}, return_tensors="pt")
    question1, answer1 = process_data(ds[0].data)
    textual_inputs = vlm.tokenizer(text=[question1], text_target= [answer1],return_tensors="pt")
    print(textual_inputs.attention_mask.shape,textual_inputs.labels.shape,textual_inputs.input_ids.shape)
    outputs = vlm(image_inputs.to(device), textual_inputs.to(device))
    print(outputs)