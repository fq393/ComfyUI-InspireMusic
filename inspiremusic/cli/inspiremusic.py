# Copyright (c) 2024 Alibaba Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import time
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from inspiremusic.cli.frontend import InspireMusicFrontEnd
from inspiremusic.cli.model import InspireMusicModel
from inspiremusic.utils.file_utils import logging
from inspiremusic.utils.utils import download_model
import torch

class InspireMusic:
    def __init__(self, model_dir, load_jit=True, load_onnx=False, dtype = "fp16", fast = False, fp16=True, hub="modelscope", repo_url=None, token=None):
        instruct = True if '-Instruct' in model_dir else False

        if model_dir is None:
            if sys.platform == "win32":
                model_dir = f"..\..\pretrained_models\{model_name}"
            else:
                model_dir = f"../../pretrained_models/{model_name}"

        logging.info(f"[DEBUG] InspireMusic.__init__ called with model_dir: {model_dir}")
        logging.info(f"[DEBUG] Checking for llm.pt at: {os.path.join(model_dir, 'llm.pt')}")
        logging.info(f"[DEBUG] llm.pt exists: {os.path.isfile(os.path.join(model_dir, 'llm.pt'))}")
        
        if not os.path.isfile(os.path.join(model_dir, "llm.pt")):
            # Extract model name from path for downloading
            model_name = os.path.basename(model_dir.rstrip('/'))
            logging.info(f"[DEBUG] Extracted model name: {model_name}")
            
            # Only attempt download if model_dir looks like a relative path or doesn't exist
            # Skip download for absolute paths that already exist (server deployment)
            if os.path.isabs(model_dir) and os.path.exists(model_dir):
                logging.info(f"[DEBUG] Model directory is absolute and exists, checking required files")
                # For server deployment, assume model files are already present
                # Just check if required files exist
                required_files = ['llm.pt', 'flow.pt', 'inspiremusic.yaml']
                logging.info(f"[DEBUG] Required files for InspireMusic: {required_files}")
                
                for file in required_files:
                    file_path = os.path.join(model_dir, file)
                    file_exists = os.path.exists(file_path)
                    logging.info(f"[DEBUG] File {file}: exists={file_exists}, path={file_path}")
                
                missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
                if missing_files:
                    logging.error(f"[DEBUG] Missing required files: {missing_files}")
                    # List all files in the model directory for debugging
                    try:
                        all_files = os.listdir(model_dir)
                        logging.info(f"[DEBUG] All files in InspireMusic model directory: {all_files}")
                    except Exception as e:
                        logging.error(f"[DEBUG] Failed to list files in InspireMusic model directory: {e}")
                    raise FileNotFoundError(f"Required model files missing in {model_dir}: {missing_files}")
            else:
                logging.info(f"[DEBUG] Attempting to download model {model_name}")
                # Download model for local/relative paths
                if hub == "modelscope":
                    from modelscope import snapshot_download
                    if model_name == "InspireMusic-Base":
                        snapshot_download(f"iic/InspireMusic", local_dir=model_dir)
                    else:
                        snapshot_download(f"iic/{model_name}", local_dir=model_dir)
                elif hub == "huggingface":
                    from huggingface_hub import snapshot_download
                    snapshot_download(repo_id=f"FunAudioLLM/{model_name}", local_dir=model_dir)
                else:
                    download_model(repo_url, model_dir, token)

        config_path = os.path.join(model_dir, 'inspiremusic.yaml')
        logging.info(f"[DEBUG] Loading config from: {config_path}")
        logging.info(f"[DEBUG] Config file exists: {os.path.exists(config_path)}")
        
        with open(config_path, 'r') as f:
            configs = load_hyperpyyaml(f)
        
        logging.info(f"[DEBUG] Config loaded successfully")
        logging.info(f"[DEBUG] Config keys: {list(configs.keys()) if isinstance(configs, dict) else 'Not a dict'}")

        # Log tokenizer configuration
        if 'get_tokenizer' in configs:
            logging.info(f"[DEBUG] get_tokenizer found in config")
        else:
            logging.error(f"[DEBUG] get_tokenizer NOT found in config")
        
        tokenizer_dir = '{}/'.format(model_dir)
        music_tokenizer_dir = '{}/music_tokenizer/'.format(model_dir)
        wavtokenizer_dir = '{}/wavtokenizer/'.format(model_dir)
        
        logging.info(f"[DEBUG] Tokenizer directory: {tokenizer_dir}")
        logging.info(f"[DEBUG] Music tokenizer directory: {music_tokenizer_dir}")
        logging.info(f"[DEBUG] Wavtokenizer directory: {wavtokenizer_dir}")
        
        # Check if directories exist
        logging.info(f"[DEBUG] Music tokenizer dir exists: {os.path.exists(music_tokenizer_dir)}")
        logging.info(f"[DEBUG] Wavtokenizer dir exists: {os.path.exists(wavtokenizer_dir)}")
        
        # List contents of tokenizer directories
        try:
            if os.path.exists(music_tokenizer_dir):
                music_files = os.listdir(music_tokenizer_dir)
                logging.info(f"[DEBUG] Music tokenizer files: {music_files}")
        except Exception as e:
            logging.error(f"[DEBUG] Failed to list music tokenizer files: {e}")
            
        try:
            if os.path.exists(wavtokenizer_dir):
                wav_files = os.listdir(wavtokenizer_dir)
                logging.info(f"[DEBUG] Wavtokenizer files: {wav_files}")
        except Exception as e:
            logging.error(f"[DEBUG] Failed to list wavtokenizer files: {e}")
        
        logging.info(f"[DEBUG] Initializing InspireMusicFrontEnd...")
        self.frontend = InspireMusicFrontEnd(configs,
                                          configs['get_tokenizer'],
                                          '{}/llm.pt'.format(model_dir),
                                          '{}/flow.pt'.format(model_dir),
                                          music_tokenizer_dir,
                                          wavtokenizer_dir,
                                          instruct,
                                          dtype,
                                          fast,
                                          fp16,
                                          configs['allowed_special'])
        logging.info(f"[DEBUG] InspireMusicFrontEnd initialized successfully")

        self.model = InspireMusicModel(configs['llm'], configs['flow'], configs['hift'], configs['wavtokenizer'], dtype, fast, fp16)
        self.model.load(os.path.join(model_dir, 'llm.pt'),
                        os.path.join(model_dir, 'flow.pt'),
                        os.path.join(model_dir, 'music_tokenizer'),
                        os.path.join(model_dir, 'wavtokenizer', "model.pt"),
                        )
        del configs

    @torch.inference_mode()
    def inference(self, task, text, audio, time_start, time_end, chorus, stream=False, sr=24000):
        if task == "text-to-music":
            for i in tqdm(self.frontend.text_normalize(text, split=True)):
                model_input = self.frontend.frontend_text_to_music(i, time_start, time_end, chorus)
                start_time = time.time()
                logging.info('prompt text {}'.format(i))
                for model_output in self.model.inference(**model_input, stream=stream):
                    music_audios_len = model_output['music_audio'].shape[1] / sr
                    logging.info('yield music len {}, rtf {}'.format(music_audios_len, (time.time() - start_time) / music_audios_len))
                    yield model_output
                    start_time = time.time()
                    
        elif task == "continuation":
            if text is None:
                if audio is not None:
                    for i in tqdm(audio):
                        model_input = self.frontend.frontend_continuation(None, i, time_start, time_end, chorus, sr, max_audio_length)
                        start_time = time.time()
                        logging.info('prompt text {}'.format(i))
                        for model_output in self.model.continuation_inference(**model_input, stream=stream):
                            music_audios_len = model_output['music_audio'].shape[1] / sr
                            logging.info('yield music len {}, rtf {}'.format(music_audios_len, (time.time() - start_time) / music_audios_len))
                            yield model_output
                            start_time = time.time()
            else:
                if audio is not None:
                    for i in tqdm(self.frontend.text_normalize(text, split=True)):
                        model_input = self.frontend.frontend_continuation(i, audio, time_start, time_end, chorus, sr, max_audio_length)
                        start_time = time.time()
                        logging.info('prompt text {}'.format(i))
                        for model_output in self.model.continuation_inference(**model_input, stream=stream):
                            music_audios_len = model_output['music_audio'].shape[1] / sr
                            logging.info('yield music len {}, rtf {}'.format(music_audios_len, (time.time() - start_time) / music_audios_len))
                            yield model_output
                            start_time = time.time()
                else:
                    print("Please input text or audio.")
        else:
            print("Currently only support text-to-music and music continuation tasks.")

    @torch.inference_mode()
    def cli_inference(self, text, audio_prompt, time_start, time_end, chorus, task, stream=False, duration_to_gen=30, sr=24000):
        if task == "text-to-music":
            model_input = self.frontend.frontend_text_to_music(text, time_start, time_end, chorus)
            logging.info('prompt text {}'.format(text))
        elif task == "continuation":
            model_input = self.frontend.frontend_continuation(text, audio_prompt, time_start, time_end, chorus, sr)
            logging.info('prompt audio length: {}'.format(len(audio_prompt)))

        start_time = time.time()
        for model_output in self.model.inference(**model_input, duration_to_gen=duration_to_gen, task=task):
            music_audios_len = model_output['music_audio'].shape[1] / sr
            logging.info('yield music len {}, rtf {}'.format(music_audios_len, (time.time() - start_time) / music_audios_len))
            yield model_output
            start_time = time.time()

    @torch.inference_mode()
    def inference_zero_shot(self, text, prompt_text, prompt_audio_16k, stream=False, sr=24000):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False)
        for i in tqdm(self.frontend.text_normalize(text, split=True)):
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_audio_16k)
            start_time = time.time()
            logging.info('prompt text {}'.format(i))
            for model_output in self.model.inference(**model_input, stream=stream):
                audio_len = model_output['music_audio'].shape[1] / sr
                logging.info('yield audio len {}, rtf {}'.format(audio_len, (time.time() - start_time) / audio_len))
                yield model_output
                start_time = time.time()
    @torch.inference_mode()
    def inference_instruct(self, text, spk_id, instruct_text, stream=False, sr=24000):
        if self.frontend.instruct is False:
            raise ValueError('{} do not support instruct inference'.format(self.model_dir))
        instruct_text = self.frontend.text_normalize(instruct_text, split=False)
        for i in tqdm(self.frontend.text_normalize(text, split=True)):
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)
            start_time = time.time()
            logging.info('prompt text {}'.format(i))
            for model_output in self.model.inference(**model_input, stream=stream):
                audio_len = model_output['music_audio'].shape[1] / sr
                logging.info('yield audio len {}, rtf {}'.format(audio_len, (time.time() - start_time) / audio_len))
                yield model_output
                start_time = time.time()
