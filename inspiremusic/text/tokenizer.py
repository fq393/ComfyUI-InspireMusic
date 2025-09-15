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

import copy
import os
import re
from typing import Iterable, List, Union
import numpy as np
import torch

from inspiremusic.text.abs_tokenizer import AbsTokenizer
from transformers import AutoTokenizer

def get_tokenizer(tokenizer_name, tokenizer_path):
    import logging
    logging.info(f"[DEBUG] get_tokenizer called with tokenizer_name='{tokenizer_name}', tokenizer_path='{tokenizer_path}'")
    
    # Check if tokenizer_path exists
    import os
    logging.info(f"[DEBUG] Tokenizer path exists: {os.path.exists(tokenizer_path)}")
    logging.info(f"[DEBUG] Tokenizer path is directory: {os.path.isdir(tokenizer_path)}")
    
    # List contents of tokenizer directory
    if os.path.exists(tokenizer_path):
        try:
            files = os.listdir(tokenizer_path)
            logging.info(f"[DEBUG] Files in tokenizer directory: {files}")
            
            # Check for specific tokenizer files
            tokenizer_model_path = os.path.join(tokenizer_path, 'tokenizer.model')
            tokenizer_json_path = os.path.join(tokenizer_path, 'tokenizer.json')
            config_json_path = os.path.join(tokenizer_path, 'config.json')
            
            logging.info(f"[DEBUG] tokenizer.model exists: {os.path.exists(tokenizer_model_path)}")
            logging.info(f"[DEBUG] tokenizer.json exists: {os.path.exists(tokenizer_json_path)}")
            logging.info(f"[DEBUG] config.json exists: {os.path.exists(config_json_path)}")
        except Exception as e:
            logging.error(f"[DEBUG] Failed to list tokenizer directory: {e}")
    
    if "qwen" in tokenizer_name:
        logging.info(f"[DEBUG] Creating QwenTokenizer with path: {tokenizer_path}")
        return QwenTokenizer(tokenizer_path,skip_special_tokens=True)
    else:
        logging.warning(f"[DEBUG] Unknown tokenizer_name: {tokenizer_name}, returning None")
        return None

class QwenTokenizer(AbsTokenizer):
    def __init__(
            self,
            token_path: str,
            skip_special_tokens: bool = True,
    ):
        import logging
        logging.info(f"[DEBUG] QwenTokenizer.__init__ called with token_path='{token_path}'")
        
        super().__init__()
        # NOTE: non-chat model, all these special tokens keep randomly initialized.
        special_tokens = {
            'eos_token': '<|endoftext|>',
            'pad_token': '<|endoftext|>',
            'additional_special_tokens': [
                '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
                '[breath]', '<strong>', '</strong>', '[noise]',
                '[laughter]', '[cough]', '[clucking]', '[accent]',
                '[quick_breath]',
            ]
        }
        
        logging.info(f"[DEBUG] Loading AutoTokenizer from: {token_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(token_path)
            logging.info(f"[DEBUG] AutoTokenizer loaded successfully")
            logging.info(f"[DEBUG] Tokenizer type: {type(self.tokenizer)}")
            if hasattr(self.tokenizer, 'name_or_path'):
                logging.info(f"[DEBUG] Tokenizer name_or_path: {self.tokenizer.name_or_path}")
        except Exception as e:
            logging.error(f"[DEBUG] AutoTokenizer.from_pretrained failed: {e}")
            logging.error(f"[DEBUG] Exception type: {type(e)}")
            import traceback
            logging.error(f"[DEBUG] Full traceback: {traceback.format_exc()}")
            raise
        
        logging.info(f"[DEBUG] Adding special tokens...")
        try:
            self.tokenizer.add_special_tokens(special_tokens)
            logging.info(f"[DEBUG] Special tokens added successfully")
        except Exception as e:
            logging.error(f"[DEBUG] Failed to add special tokens: {e}")
            raise
        
        self.skip_special_tokens = skip_special_tokens
        logging.info(f"[DEBUG] QwenTokenizer initialization completed")

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def text2tokens(self, line: str) -> List:
        tokens = self.tokenizer([line], return_tensors="pt")
        tokens = tokens["input_ids"][0].cpu().tolist()
        return tokens

    def tokens2text(self, tokens) -> str:
        tokens = torch.tensor(tokens, dtype=torch.int64)
        text = self.tokenizer.batch_decode([tokens], skip_special_tokens=self.skip_special_tokens)[0]
        return text



def get_qwen_vocab_size(token_type: str):
    if "qwen1.5" in token_type.lower() or "qwen2.0" in token_type.lower() or "qwen2.5" in token_type.lower():
        # 293 for special and extra tokens, including endoftext, im_start, im_end, endofprompt and others in the future.
        # model.vocab_size = 151936, tokenizer.vocab_size = 151643
        # NOTE: the first three special tokens (endoftext, im_start, im_end) are trained in Chat series models,
        # others are kept in random initialization state.
        return 151643 + 293
    else:
        raise ValueError(f"Unknown tokenizer {token_type}")