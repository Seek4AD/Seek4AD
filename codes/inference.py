#If accessing via API, you can skip this step and directly use the inference_with_api function.
import torch
import os
import sys
utils_path = "/home/Seek4AD/"
sys.path.append(utils_path) 
from argparse import ArgumentParser
import sys
from PIL import Image
from transformers import  AutoProcessor,TextIteratorStreamer
from transformers import AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration
from adseeker.search import searchengine
from seek_agent.tools.single_test import seek_agent
from ad_expert.test_single_image import process_image
import argparse
from peft import LoraConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from threading import Thread
def search_agent(img):
    parser = ArgumentParser()
    parser.add_argument('--config',default="./seek_agent/model/efficientnetv2/efficientnetv2_b0.py" ,help='Config file')
    parser.add_argument('--classes-map', default='seek_agent/datas/annotations.txt', help='classes map of datasets')
    parser.add_argument('--device', default='cpu', help='Device used for inference')
    parser.add_argument('--save-path',help='The path to save prediction image, default not to save.')
    args_search = parser.parse_args()
    print("img is",img)
    return seek_agent(args_search,img)
    
def I_IRAG(img,gmm=True,input_gmm=20):
    #search_engine = SearchEngine(dataset='adseek', node_dir_prefix='colqwen_ingestion',embed_model_name='vidore/colqwen2-v1.0')
    #search_engine.gmm =gmm
    #search_engine.input_gmm = input_gmm
    #recall_results = search_engine.search(search_agent(img))
    seeker=search_agent(img)
    recall_results=searchengine(seeker)
    
    return recall_results["source_nodes"][0]["node"]["metadata"]["file_path"]

    
def AD_expert_mask(input_img):
   img=Image.open(input_img)
   category_tex=search_agent(input_img)
   out_img,out_score=process_image(img,category_tex)
   out_img_url="output_expert.png"
   out_img = Image.fromarray(out_img)
   out_img.save(out_img_url)
   return out_img_url


def inference(img_url, prompt, system_prompt="You are a helpful assistant", max_new_tokens=32000):
  
  

  
  if args.seek_agent==False:
      messages = [
        {
          "role": "system",
          "content": system_prompt
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt
            },
            {
               "type":"image" ,
              "image": img_url
            }
          ]
        }
        ]
  else:
      messages = [
        {
          "role": "system",
          "content": system_prompt
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt
            },
            {
              "type":"image",
              "image": f"file://{img_url[0]}" 
            },
            {
              "type":"image",
              "image": f"file://{img_url[1]}" 
            }
          ]
        }
        ]

  images,videos=process_vision_info(messages)
  text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  print(text)
  print("input:\n",text)
  inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to('cuda')

  device = next(model.parameters()).device
        
  tokenizer = processor.tokenizer
  streamer = TextIteratorStreamer(tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)

  gen_kwargs = {'max_new_tokens': 2048, 'streamer': streamer, **inputs}
  thread = Thread(target=model.generate, kwargs=gen_kwargs)
  thread.start()

  output_text = ''
  for i,new_text in enumerate(streamer):
      output_text += new_text
                

  input_height = inputs['image_grid_thw'][0][1]*14
  input_width = inputs['image_grid_thw'][0][2]*14

  return output_text, input_height, input_width
if __name__ == "__main__":
    parser = argparse.ArgumentParser("seek4ad", add_help=True)

    # Paths and configurations
    parser.add_argument("--ckt_path", type=str, default="/home/datasets/Qwen2.5-VL-7B-Instruct/",
                        help="Path to the pre-trained model ")
    parser.add_argument("--img_path", type=str, default="/home/datasets/mmad/MMAD/DS-MVTec/carpet/image/cut/001.png",
                        help="Path to the test anomaly image ")
    parser.add_argument("--use_pretrain", type=bool, default=True,
                        help="whether use the prtrained model ")
    parser.add_argument("--train_path", type=str, default="output/Qwen2.5-VL-LoRA/checkpoint-XXX",
                        help="Path to the Lora-trained model ")
    parser.add_argument("--use_expert", type=bool, default=False,
                        help="whether use the ad_expert model to help inference")
    parser.add_argument("--seek_agent", type=bool, default=True,
                        help="whether use the ad_expert seek_agent to help inference" )
    parser.add_argument("--IRAG", type=bool, default=True,
                        help="whether use the I-Irag to help inference" )
    parser.add_argument("--database_path",type=bool,default="/home/Seek4AD/adseeker/data/ExampleDataset/adseek/"
                        , help="The location where the rag knowledge base is stored")

    args = parser.parse_args()



    if args.use_pretrain: 
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.ckt_path, device_map="auto")
        processor = AutoProcessor.from_pretrained(args.ckt_path) 
    else: 
        config = LoraConfig(
          task_type="CAUSAL_LM",
          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
          inference_mode=False,
          r=64,
          lora_alpha=16,
          lora_dropout=0.05,
          bias="none",
        ) 
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.ckt_path, device_map="auto")
        processor = AutoProcessor.from_pretrained(args.ckt_path) 
        tokenizer = AutoTokenizer.from_pretrained(args.ckt_path)  
        model = PeftModel.from_pretrained(model, args.train_path, config=config)
    


    image = Image.open(args.img_path)
    system_prompt="You are an AI specialized in recognizing and extracting text from images. Your mission is to analyze the image document and generate the result in QwenVL Document Parser HTML format using specified tags while maintaining user privacy and data integrity."
    prompt =  "您好，请问图中有什么缺陷"
    if args.use_expert==False and args.seek_agent==False and args.IRAG==False: 
      ## Use a local HuggingFace model to inference.
        output, input_height, input_width = inference(args.img_path, prompt,system_prompt)
    elif args.use_expert==False and args.seek_agent==True and args.IRAG==True:
        knowledge_image_url=I_IRAG(args.img_path)
        img_path=[args.img_path,knowledge_image_url]

        output, input_height, input_width = inference(img_path, prompt,system_prompt)
    elif args.use_expert==True and args.seek_agent==True and args.IRAG==True:
        knowledge_image_url=I_IRAG(args.img_path)
        expert_img_url=AD_expert_mask(args.img_path)
        img_path=[expert_img_url,knowledge_image_url]
        output, input_height, input_width = inference(img_path, prompt,system_prompt)
    elif args.use_expert==True and args.seek_agent==True and args.IRAG==False:
        expert_img_url=AD_expert_mask(args.img_path)
        img_path=[expert_img_url,args.img_path]
        output, input_height, input_width = inference(img_path, prompt,system_prompt)
    else:
        raise ValueError("not permission")
        





    # Visualization
    print(input_height, input_width)
    print(output)


