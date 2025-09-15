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
from functools import partial
import torch
from typing import Callable
import re
import inflect
from inspiremusic.cli.model import InspireMusicModel
from inspiremusic.utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph
from inspiremusic.wavtokenizer.decoder.pretrained import WavTokenizer

class InspireMusicFrontEnd:
    def __init__(self,
                 configs: Callable,
                 get_tokenizer: Callable,
                 llm_model: str,
                 flow_model: str,
                 music_tokenizer_dir: str,
                 audio_tokenizer_dir: str,
                 instruct: bool = False,
                 dtype: str = "fp16",
                 fast: bool = False,
                 fp16: bool = True,
                 allowed_special: str = 'all'):
        import logging
        logging.info(f"[DEBUG] InspireMusicFrontEnd.__init__ called")
        logging.info(f"[DEBUG] Parameters: llm_model={llm_model}, flow_model={flow_model}")
        logging.info(f"[DEBUG] music_tokenizer_dir={music_tokenizer_dir}")
        logging.info(f"[DEBUG] audio_tokenizer_dir={audio_tokenizer_dir}")
        
        logging.info(f"[DEBUG] Calling get_tokenizer function...")
        try:
            self.tokenizer = get_tokenizer()
            logging.info(f"[DEBUG] get_tokenizer() returned: {type(self.tokenizer)}")
            if hasattr(self.tokenizer, 'name_or_path'):
                logging.info(f"[DEBUG] Tokenizer name_or_path: {self.tokenizer.name_or_path}")
        except Exception as e:
            logging.error(f"[DEBUG] get_tokenizer() failed: {e}")
            raise
        
        self.audio_tokenizer_dir = audio_tokenizer_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"[DEBUG] Using device: {self.device}")

        self.bandwidth_id = torch.tensor([0]).to(self.device)
        
        logging.info(f"[DEBUG] Loading WavTokenizer from: {audio_tokenizer_dir}")
        try:
            self.wavtokenizer = WavTokenizer.from_pretrained_feat(f"{audio_tokenizer_dir}/config.yaml", f"{audio_tokenizer_dir}/model.pt").to(self.device)
            logging.info(f"[DEBUG] WavTokenizer loaded successfully")
        except Exception as e:
            logging.error(f"[DEBUG] WavTokenizer loading failed: {e}")
            raise

        logging.info(f"[DEBUG] Creating InspireMusicModel...")
        try:
            self.model = InspireMusicModel(configs['llm'], configs['flow'], configs['hift'], configs['wavtokenizer'], dtype, fast, fp16)
            logging.info(f"[DEBUG] InspireMusicModel created successfully")
        except Exception as e:
            logging.error(f"[DEBUG] InspireMusicModel creation failed: {e}")
            raise
        
        logging.info(f"[DEBUG] Loading model weights...")
        try:
            self.model = self.model.load(llm_model, flow_model, music_tokenizer_dir, audio_tokenizer_dir)
            logging.info(f"[DEBUG] Model weights loaded successfully")
        except Exception as e:
            logging.error(f"[DEBUG] Model weights loading failed: {e}")
            raise

        self.instruct = instruct
        self.allowed_special = allowed_special
        self.inflect_parser = inflect.engine()
        logging.info(f"[DEBUG] InspireMusicFrontEnd initialization completed")

    def _extract_text_token(self, text):
        text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
        text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
        text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)
        return text_token, text_token_len

    def _extract_audio_token(self, audio, sample_rate=24000):
        audio = torch.tensor(audio, dtype=torch.float32, device=self.device)
        _, audio_token = self.wavtokenizer.encode_infer(audio, bandwidth_id=self.bandwidth_id)
        audio_token = audio_token.squeeze(0)
        audio_token_len = torch.tensor([audio_token.shape[1]], dtype=torch.int32, device=self.device)
        return audio_token, audio_token_len

    def text_normalize(self, text, split=True):
        text = text.strip()
        if contains_chinese(text):
            text = text.replace("\n", "")
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "、")
            text = text.replace(" - ", "，")
            text = remove_bracket(text)
            text = re.sub(r'[，,]+$', '。', text)
            texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "zh", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False))
        else:
            text = spell_out_number(text, self.inflect_parser)
            texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "en", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False))
        if split is False:
            return text
        return texts

    def frontend_text_to_music(self, text, time_start, time_end, chorus):
        text_token, text_token_len = self._extract_text_token(text)
        model_input = {"text": text, "audio_token": None, "audio_token_len": None,
                                "text_token": text_token, "text_token_len": text_token_len,
                                "embeddings": [time_start, time_end, chorus], "raw_text":text}
        return model_input

    def frontend_continuation(self, text, audio, time_start, time_end, chorus, target_sr=24000):
        if text is None:
            text_token = None
            text_token_len = None
        else:
            text_token, text_token_len = self._extract_text_token(text)
        audio_token, audio_token_len = self._extract_audio_token(audio, target_sr)
        model_input = {"text": text, "audio_token": audio_token, "audio_token_len": audio_token_len,
                                "text_token": text_token, "text_token_len": text_token_len,
                                "embeddings": [time_start, time_end, chorus], "raw_text":text}
        return model_input

