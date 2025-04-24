
import torch
import json
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
import sys
import PIL
from PIL import Image

from transformers import AutoProcessor, TextIteratorStreamer
import seaborn as sns
import time
import random
import re
import pandas as pd
import cv2
from threading import Thread
utils_path = os.path.join(os.path.dirname(__file__), '_modeling_qwen2_5_vl')
sys.path.append(utils_path)
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import AutoTokenizer
import argparse
from helper.summary import caculate_accuracy_mmad
from GPT4.gpt4v import GPT4Query, instruction
from peft import LoraConfig
from peft import PeftModel
from adseeker.search_engine import SearchEngine
from ad_expert.test_single_image import process_image
from seek_agent.utils.train_utils import get_info, file2dict
from argparse import ArgumentParser
from seek_agent.utils.inference import init_model,inference_model
from torch.nn.parallel import DataParallel
from seek_agent.model.build import BuildNet
from seek_agent.utils.train_utils import get_info, file2dict, set_random_seed
class Qwen2Query(GPT4Query):
     



     def __init__(self, image_path, text_gt, processor, model, few_shot=[], visualization=False, args=None):
        super(Qwen2Query, self).__init__(image_path, text_gt, few_shot, visualization)
        self.processor = processor
        self.model = model
        self.args = args
        
        






     def generate_answer(self):
        questions, answers = self.parse_conversation(self.text_gt)
        if questions == [] or answers == []:
            return questions, answers, None
        
        history = []
        gpt_answers = []
        for i in range(len(questions)):
            part_questions = questions[i:i + 1]
            prompt,image_paths=self.get_query(part_questions)
            
            text= processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            
            images,videos=process_vision_info(prompt)
            
            
            #images = []
            #for path in image_paths:
                #print(path)
                #try:
                    #images.append(Image.open(path))
                #except PIL.UnidentifiedImageError:
                    #print(f"Skipped invalid image: {path}")
            device = next(self.model.parameters()).device  # 获取模型所在的设备
            inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt").to(device)
            tokenizer = processor.tokenizer
            streamer = TextIteratorStreamer(tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)

            gen_kwargs = {'max_new_tokens': 2048, 'streamer': streamer, **inputs}
            thread = Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()

            output_text = ''
            for i,new_text in enumerate(streamer):
                output_text += new_text
            print(output_text)
                

                

            #output_ids=self.model.generate(**inputs, max_new_tokens=1024)
            #generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            #output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if self.args.record_history:
                history.append((part_questions[0]['text'], output_text))
            
            gpt_answer = self.parse_answer(output_text, part_questions[0]['options'])
            gpt_answers.append(gpt_answer[0])
        return questions,answers,gpt_answers
     
    
     def I_IRAG(self,img,gmm=True,input_gmm=20):
         #search_engine = SearchEngine(dataset='adseek', node_dir_prefix='colqwen_ingestion',embed_model_name='vidore/colqwen2-v1.0')
         #search_engine.gmm =gmm
         #search_engine.input_gmm = input_gmm
         #recall_results = search_engine.search(search_agent(img))
         seeker=self.search_agent(img)
         
         recall_results=search_engine.search(seeker)
        
         return recall_results["source_nodes"][0]["node"]["metadata"]["file_path"]

    
     def AD_expert_mask(self,input_img):
         img=Image.open(input_img)
         category_tex=self.search_agent(input_img)
         out_img,out_score=process_image(img,category_tex)
         out_img_url="output_expert.png"
         out_img = Image.fromarray(out_img)
         out_img.save(out_img_url)
         return out_img_url
     def search_agent(self,img):
        
        
        try:
            image = Image.open(img)
            if image.format == 'PNG':
                image = image.convert('RGB')
                new_img_path = img.rsplit('.', 1)[0] + '.jpg'
                image.save(new_img_path, 'JPEG')
                img = new_img_path
        except Exception as e:
            print(f"图片转换出错: {e}")
        
        
        result = inference_model(model_agent, img, val_pipeline, classes_names,label_names)
        return result["pred_class"]
     def get_query(self, conversation):
         incontext = []
         
         if self.few_shot:
             incontext.append({
                "text": f"Following is/are {len(self.few_shot)} image of normal sample, which can be used as a template to compare the image being queried."
             })
             
         for ref_image_path in self.few_shot:
            # if not is_anomaly:
                
                ref_image = cv2.imread(ref_image_path)
               
                
                
                if self.visualization:
                    self.visualize_image(ref_image)
                #ref_base64_image = self.encode_image_to_base64(ref_image)
                incontext.append({
                    "image": ref_image_path
                    })
         if self.args.seek_agent==True and self.args.IRAG==True:
             knowledge_image_url=self.I_IRAG(img=self.image_path)
             ref_image_path=knowledge_image_url
         if self.args.use_expert==True:
             expert_url=self.AD_expert_mask(self.image_path)
         image = cv2.imread(self.image_path)
         image_paths=[]
         image_paths.append(ref_image_path)
         image_paths.append(self.image_path)
         
         print(image_paths)
         
         payload2 = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": instruction
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Following is/are {len(self.few_shot)} picture of normal, which can be used as a object reference."
            },
            
            {
                "type": "image",
                "image": f"file://{ref_image_path}"
            },
            {
                "type": "text",
                "text": "Following is the query Defect Heat Map image:"
            },
            {
                "type": "image",
                "image": f"file://{self.image_path}"
            },
            {
                "type": "text",
                "text": "Following is the question list:"
            },
            {
                "type": "text",
                "text": conversation
            },
            {
                "type": "text",
                "text": "Only one of the four options (A, B, C, D) should be given as the answer"
            }
           ]
            }
         ]
         return payload2,image_paths

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/datasets/Qwen2.5-VL-3B-Instruct/")
    parser.add_argument("--use_pretrain", type=bool, default=False,
                        help="whether use the prtrained model ")
    parser.add_argument("--train_path", type=str, default="/home/datasets/Qwen2.5-VL-LoRA1-5epoch/checkpoint-15/",
                        help="Path to the Lora-trained model ")
    parser.add_argument("--few_shot_model", type=int, default=1)
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--similar_template", action="store_true")
    parser.add_argument("--record_history", action="store_true")
    parser.add_argument("--use_expert", type=bool, default=False,
                        help="whether use the ad_expert model to help inference")
    parser.add_argument("--seek_agent", type=bool, default=False,
                        help="whether use the ad_expert seek_agent to help inference" )
    parser.add_argument("--IRAG", type=bool, default=False,
                        help="whether use the I-Irag to help inference" )
    parser.add_argument('--config',default="./seek_agent/model/efficientnetv2/efficientnetv2_b0.py" ,help='Config file')
    parser.add_argument('--classes-map', default='seek_agent/datas/annotations.txt', help='classes map of datasets')
    parser.add_argument('--device', default='cuda', help='Device used for inference')
    parser.add_argument('--gpu_id', default="", help='Device used for inference')
    parser.add_argument('--save-path',help='The path to save prediction image, default not to save.')
    
    args = parser.parse_args()
    model_path=args.model_path
    if args.seek_agent==True:
        
        classes_names, label_names = get_info(args.classes_map)
        # build the model from a config file and a checkpoint file
        model_cfg, train_pipeline, val_pipeline,data_cfg, lr_config, optimizer_cfg = file2dict(args.config)
        if args.device is not None:
            device = torch.device(args.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model_agent = BuildNet(model_cfg)
        model_agent = init_model(model_agent, data_cfg, device=device, mode='eval')
    if args.IRAG==True:
        search_engine = SearchEngine(dataset='adseek', node_dir_prefix='colqwen_ingestion',embed_model_name='vidore/colqwen2-v1.0')
        # Set parameters of dynamic retriever
        search_engine.gmm = True
        search_engine.input_gmm = 20 # The default setting is K
    


   
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16,low_cpu_mem_usage=True  )
    processor = AutoProcessor.from_pretrained(model_path)
    if args.use_pretrain==False:
        config = LoraConfig(
          task_type="CAUSAL_LM",
          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
          inference_mode=False,
          r=64,
          lora_alpha=16,
          lora_dropout=0.05,
          bias="none",
        ) 
        tokenizer = AutoTokenizer.from_pretrained(model_path)  
        model = PeftModel.from_pretrained(model, args.train_path, config=config)
    processor.image_processor.size = {"height": 600, "width":600}  # 将图像调整为448x448像素
    
    model_name = os.path.split(model_path.rstrip('/'))[-1]
    if args.similar_template:
        model_name = model_name + "_Similar_template"
    answers_json_path = f"result/answers_{args.few_shot_model}_shot_{model_name}_mmad_3B——5epoch.json"
    if not os.path.exists("result"):
        os.makedirs("result")
    print(f"Answers will be saved at {answers_json_path}")
    # 用于存储所有答案
    if os.path.exists(answers_json_path):
        with open(answers_json_path, "r") as file:
            all_answers_json = json.load(file)
    else:
        all_answers_json = []

    existing_images = [a["image"] for a in all_answers_json]

    cfg = {
        "data_path": "/home/datasets/mmad/MMAD/",
        "json_path": "/home/datasets/mmad/MMAD/mmad-ori.json"
    }
    args.data_path = cfg["data_path"]

    with open(cfg["json_path"], "r") as file:
        chat_ad = json.load(file)

    for image_path in tqdm(chat_ad.keys()):
        if image_path in existing_images and not args.reproduce:
            continue
        text_gt = chat_ad[image_path]
        if args.similar_template:
            few_shot = text_gt["similar_templates"][:args.few_shot_model]
        else:
            few_shot = text_gt["random_templates"][:args.few_shot_model]

        rel_image_path = os.path.join(args.data_path, image_path)
        
        
        
        rel_few_shot = [os.path.join(args.data_path, path) for path in few_shot]
        
        qwenquery = Qwen2Query(image_path=rel_image_path, text_gt=text_gt,
                           processor=processor, model=model, few_shot=rel_few_shot, visualization=False, args=args)
        questions, answers, gpt_answers = qwenquery.generate_answer()
        if gpt_answers is None or len(gpt_answers) != len(answers):
            print(f"Error at {image_path}")
            continue
        correct = 0
        for i, answer in enumerate(answers):
            if gpt_answers[i] == answer:
                correct += 1
        accuracy = correct / len(answers)
        print(f"Accuracy: {accuracy:.2f}")

        questions_type = [conversion["type"] for conversion in text_gt["conversation"]]
        # 更新答案记录
        for q, a, ga, qt in zip(questions, answers, gpt_answers, questions_type):
            answer_entry = {
                "image": image_path,
                "question": q,
                "question_type": qt,
                "correct_answer": a,
                "gpt_answer": ga
            }

            all_answers_json.append(answer_entry)

        # 保存答案为JSON
        with open(answers_json_path, "w") as file:
            json.dump(all_answers_json, file, indent=4)

    caculate_accuracy_mmad(answers_json_path)
    


