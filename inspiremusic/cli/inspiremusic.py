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

# Use official matcha-tts package (no additional path setup needed)

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

        if not os.path.isfile(os.path.join(model_dir, "llm.pt")):
            # Extract model name from path for downloading
            model_name = os.path.basename(model_dir.rstrip('/'))
            
            # Only attempt download if model_dir looks like a relative path or doesn't exist
            # Skip download for absolute paths that already exist (server deployment)
            if os.path.isabs(model_dir) and os.path.exists(model_dir):
                # For server deployment, assume model files are already present
                # Just check if required files exist
                # Note: tokenizer.model is not required for HuggingFace tokenizers
                required_files = ['llm.pt', 'flow.pt', 'inspiremusic.yaml']
                
                missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
                if missing_files:
                    logging.error(f"Missing required files: {missing_files}")
                    # List all files in the model directory for debugging
                    try:
                        all_files = os.listdir(model_dir)
                        logging.info(f"All files in model directory: {all_files}")
                    except Exception as e:
                        logging.error(f"Failed to list files in model directory: {e}")
                    raise FileNotFoundError(f"Required model files missing in {model_dir}: {missing_files}")
                
                # Check if at least one tokenizer format is available
                tokenizer_formats = ['tokenizer.model', 'tokenizer.json']
                has_tokenizer = any(os.path.exists(os.path.join(model_dir, f)) for f in tokenizer_formats)
                if not has_tokenizer:
                    logging.warning(f"No tokenizer files found, but continuing as HuggingFace AutoTokenizer may handle this")
            else:
                logging.info(f"Downloading model {model_name}")
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

        configs = self._create_default_config(model_dir)

        # Log tokenizer configuration
        if 'get_tokenizer' not in configs:
            logging.error(f"get_tokenizer NOT found in config")
        
        tokenizer_dir = '{}/'.format(model_dir)
        music_tokenizer_dir = '{}/music_tokenizer/'.format(model_dir)
        wavtokenizer_dir = '{}/wavtokenizer/'.format(model_dir)
        
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

        self.model = InspireMusicModel(configs['llm'], configs['flow'], configs['hift'], configs['wavtokenizer'], dtype, fast, fp16)
        self.model.load(os.path.join(model_dir, 'llm.pt'),
                        os.path.join(model_dir, 'flow.pt'),
                        os.path.join(model_dir, 'music_tokenizer'),
                        os.path.join(model_dir, 'wavtokenizer', "model.pt"),
                        )
        del configs

    def _create_default_config(self, model_dir):
        """创建默认配置，避免读取 YAML 配置文件"""
        from inspiremusic.text.tokenizer import get_tokenizer
        from inspiremusic.llm.llm import LLM
        from inspiremusic.flow.flow import MaskedDiff
        from inspiremusic.hifigan.generator import HiFTGenerator
        
        # 导入必要的模块
        from inspiremusic.transformer.qwen_encoder import QwenEmbeddingEncoder
        from inspiremusic.transformer.encoder import ConformerEncoder
        from inspiremusic.flow.length_regulator import InterpolateRegulator
        from inspiremusic.flow.flow_matching import ConditionalCFM
        from inspiremusic.flow.decoder import ConditionalDecoder
        from inspiremusic.hifigan.f0_predictor import ConvRNNF0Predictor
        from inspiremusic.utils.common import topk_sampling
        from omegaconf import DictConfig
        import functools
        import torch
        from librosa.filters import mel as librosa_mel_fn
        
        # 定义 mel_spectrogram 函数
        def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
            """Mel spectrogram function from matcha.utils.audio"""
            mel_basis = {}
            hann_window = {}
            
            if torch.min(y) < -1.0:
                print("min value is ", torch.min(y))
            if torch.max(y) > 1.0:
                print("max value is ", torch.max(y))

            if f"{str(fmax)}_{str(y.device)}" not in mel_basis:
                mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
                mel_basis[str(fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
                hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

            y = torch.nn.functional.pad(
                y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
            )
            y = y.squeeze(1)

            spec = torch.view_as_real(
                torch.stft(
                    y,
                    n_fft,
                    hop_length=hop_size,
                    win_length=win_size,
                    window=hann_window[str(y.device)],
                    center=center,
                    pad_mode="reflect",
                    normalized=False,
                    onesided=True,
                    return_complex=True,
                )
            )

            spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
            spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
            
            # spectral_normalize_torch
            spec = torch.log(torch.clamp(spec, min=1e-5) * 1)
            
            return spec
        
        # 创建 QwenEmbeddingEncoder
        qwen_encoder = QwenEmbeddingEncoder(
            input_size=512,
            pretrain_path=model_dir
        )
        
        # 创建 sampling 函数
        sampling_func = functools.partial(topk_sampling, top_k=350)
        
        # 创建 ConformerEncoder
        conformer_encoder = ConformerEncoder(
            output_size=512,
            attention_heads=4,
            linear_units=1024,
            num_blocks=3,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
            input_layer='linear',
            pos_enc_layer_type='rel_pos_espnet',
            selfattention_layer_type='rel_selfattn',
            input_size=256,
            use_cnn_module=False,
            macaron_style=False
        )
        
        # 创建 InterpolateRegulator
        length_regulator = InterpolateRegulator(
            channels=512,
            sampling_ratios=[1, 1, 1, 1]
        )
        
        # 创建 ConditionalDecoder
        conditional_decoder = ConditionalDecoder(
            in_channels=1024,
            out_channels=512,
            channels=[256, 256],
            dropout=0.0,
            attention_head_dim=64,
            n_blocks=4,
            num_mid_blocks=8,
            num_heads=8,
            act_fn='gelu'
        )
        
        # 创建 ConditionalCFM
        cfm_params = DictConfig({
            'sigma_min': 1e-06,
            'solver': 'euler',
            't_scheduler': 'cosine',
            'training_cfg_rate': 0.2,
            'inference_cfg_rate': 0.7,
            'reg_loss_type': 'l1'
        })
        
        conditional_cfm = ConditionalCFM(
            in_channels=240,
            cfm_params=cfm_params,
            estimator=conditional_decoder
        )
        
        # 创建 ConvRNNF0Predictor
        f0_predictor = ConvRNNF0Predictor(
            num_class=1,
            in_channels=80,
            cond_channels=512
        )
        
        # 创建实际的模型对象
        llm_config = {
            'text_encoder_input_size': 512,
            'llm_input_size': 1536,
            'llm_output_size': 1536,
            'audio_token_size': 4096,
            'llm': qwen_encoder,
            'sampling': sampling_func,
            'length_normalized_loss': True,
            'lsm_weight': 0,
            'text_encoder_conf': {'name': 'none'},
            'train_cfg_ratio': 0.2,
            'infer_cfg_ratio': 3.0
        }
        
        flow_config = {
            'input_size': 256,
            'output_size': 80,
            'output_type': 'mel',
            'vocab_size': 4096,
            'input_frame_rate': 75,
            'only_mask_loss': True,
            'encoder': conformer_encoder,
            'length_regulator': length_regulator,
            'decoder': conditional_cfm,
            'generator_model_dir': os.path.join(model_dir, 'music_tokenizer')
        }
        
        hift_config = {
            'in_channels': 80,
            'base_channels': 512,
            'nb_harmonics': 8,
            'sampling_rate': 24000,
            'nsf_alpha': 0.1,
            'nsf_sigma': 0.003,
            'nsf_voiced_threshold': 10,
            'upsample_rates': [8, 8],
            'upsample_kernel_sizes': [16, 16],
            'resblock_kernel_sizes': [3, 7, 11],
            'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            'source_resblock_kernel_sizes': [7, 11],
            'source_resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5]],
            'lrelu_slope': 0.1,
            'audio_limit': 0.99,
            'f0_predictor': f0_predictor
        }
        
        # 创建默认配置字典
        configs = {
            # 基础参数
            'sample_rate': 24000,
            'text_encoder_input_size': 512,
            'llm_input_size': 1536,
            'llm_output_size': 1536,
            'basemodel_path': model_dir,
            'generator_path': os.path.join(model_dir, 'music_tokenizer'),
            
            # 创建实际的模型对象
            'llm': LLM(**llm_config),
            'flow': MaskedDiff(**flow_config),
            'hift': HiFTGenerator(**hift_config),
            'wavtokenizer': {},  # WavTokenizer 将在后续加载
            
            # Tokenizer 配置
            'allowed_special': 'all',
            
            # 添加 mel_spectrogram 函数
            'mel_spectrogram': mel_spectrogram
        }
        
        # 设置 get_tokenizer 函数，使用 functools.partial 来预设参数
        import functools
        configs['get_tokenizer'] = functools.partial(
            get_tokenizer,
            tokenizer_path=model_dir,
            tokenizer_name='qwen-2.5'
        )
        

        
        return configs

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
