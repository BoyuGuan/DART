# -*- coding: utf-8 -*-
"""
Custom Trainer for computing translation-specific loss alongside the standard loss.
"""

import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, Optional, Union, Any
import re


class TranslationLossTrainer(Trainer):
    """
    自定义Trainer，额外计算翻译部分的loss并记录到wandb。
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        重写compute_loss方法，计算总loss和翻译部分的loss。
        """
        # 标准的forward pass计算总loss
        outputs = model(**inputs)
        total_loss = outputs.loss
        
        # 计算翻译部分的loss
        translation_loss = self._compute_translation_loss(model, inputs, outputs)
        
        # 记录到wandb
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "translation_loss": translation_loss.item() if translation_loss is not None else 0.0,
                "total_loss": total_loss.item(),
            })
        
        if return_outputs:
            return total_loss, outputs
        return total_loss
    
    def _compute_translation_loss(self, model, inputs, outputs) -> Optional[torch.Tensor]:
        """
        计算翻译部分的loss。
        
        从GPT响应中提取"The translation is:"或"So the translation is:"之后的内容，
        只计算这部分的loss。
        """
        try:
            # 获取必要的输入
            input_ids = inputs.get("input_ids")
            labels = inputs.get("labels")
            attention_mask = inputs.get("attention_mask", None)
            
            if input_ids is None or labels is None:
                return None
            
            # 获取模型的logits
            logits = outputs.logits
            
            # 创建翻译部分的mask
            translation_mask = self._create_translation_mask(input_ids, labels)
            
            if translation_mask is None or translation_mask.sum() == 0:
                return None
            
            # 计算翻译部分的loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_mask = translation_mask[..., 1:].contiguous()
            
            # Flatten the tokens
            batch_size, seq_length, vocab_size = shift_logits.shape
            flat_logits = shift_logits.view(-1, vocab_size)
            flat_labels = shift_labels.view(-1)
            flat_mask = shift_mask.view(-1)
            
            # 计算所有位置的loss（不做reduction）
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            flat_loss = loss_fct(flat_logits, flat_labels)
            
            # 只保留翻译部分的loss
            masked_loss = flat_loss * flat_mask.float()
            
            # 计算平均loss
            if flat_mask.sum() > 0:
                translation_loss = masked_loss.sum() / flat_mask.sum()
            else:
                translation_loss = torch.tensor(0.0, device=flat_loss.device)
            
            return translation_loss
            
        except Exception as e:
            # 如果计算失败，返回None
            print(f"Warning: Failed to compute translation loss: {str(e)}")
            return None
    
    def _create_translation_mask(self, input_ids: torch.Tensor, labels: torch.Tensor) -> Optional[torch.Tensor]:
        """
        创建翻译部分的mask。
        
        策略：查找"The translation is:"或"So the translation is:"之后的token，
        将这些位置标记为1，其余为0。
        """
        try:
            batch_size, seq_length = labels.shape
            translation_mask = torch.zeros_like(labels, dtype=torch.bool)
            
            # 获取tokenizer（从model的config中）
            # 这里我们需要解码文本来找到翻译起始位置
            
            # 定义翻译标记的可能形式
            translation_markers = [
                "The translation is:",
                "So the translation is:",
                "translation is:",
            ]
            
            for batch_idx in range(batch_size):
                # 获取当前样本的labels（排除padding）
                sample_labels = labels[batch_idx]
                valid_labels = sample_labels[sample_labels != -100]
                
                if len(valid_labels) == 0:
                    continue
                
                # 尝试解码来找到翻译起始位置
                # 注意：这里简化处理，假设翻译部分在最后
                # 更精确的方法需要解码整个序列
                
                # 从后往前找，假设翻译是最后一个有效标签的连续部分
                # 由于CoT格式，翻译通常在最后，我们标记最后20%的有效token作为翻译部分
                valid_indices = (sample_labels != -100).nonzero(as_tuple=True)[0]
                
                if len(valid_indices) > 0:
                    # 标记最后30%的有效token作为翻译部分
                    # 这是一个启发式方法，可以根据实际情况调整
                    num_valid = len(valid_indices)
                    translation_start_idx = int(num_valid * 0.7)  # 从70%位置开始
                    
                    for idx in valid_indices[translation_start_idx:]:
                        translation_mask[batch_idx, idx] = True
            
            return translation_mask
            
        except Exception as e:
            print(f"Warning: Failed to create translation mask: {str(e)}")
            return None


class TranslationLossTrainerV2(Trainer):
    """
    改进版本：使用tokenizer解码来精确定位翻译部分。
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 保存tokenizer引用
        self._processing_class = kwargs.get('processing_class') or kwargs.get('tokenizer')
        # 用于临时存储translation_loss
        self._current_translation_loss = None
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        重写compute_loss方法，计算总loss和翻译部分的loss。
        """
        # 标准的forward pass计算总loss
        outputs = model(**inputs)
        total_loss = outputs.loss
        
        # 计算翻译部分的loss
        translation_loss = self._compute_translation_loss(model, inputs, outputs)
        
        # 存储translation_loss，稍后在log中一起输出
        if translation_loss is not None:
            self._current_translation_loss = translation_loss.item()
        
        if return_outputs:
            return total_loss, outputs
        return total_loss
    
    def _compute_translation_loss(self, model, inputs, outputs) -> Optional[torch.Tensor]:
        """
        计算翻译部分的loss（改进版，使用tokenizer解码）。
        """
        try:
            input_ids = inputs.get("input_ids")
            labels = inputs.get("labels")
            
            if input_ids is None or labels is None:
                return None
            
            logits = outputs.logits
            
            # 创建翻译部分的mask
            translation_mask = self._create_translation_mask_v2(input_ids, labels)
            
            if translation_mask is None or translation_mask.sum() == 0:
                return None
            
            # 计算loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_mask = translation_mask[..., 1:].contiguous()
            
            vocab_size = shift_logits.shape[-1]
            flat_logits = shift_logits.view(-1, vocab_size)
            flat_labels = shift_labels.view(-1)
            flat_mask = shift_mask.view(-1)
            
            # 只对非-100的标签计算loss
            valid_positions = flat_labels != -100
            combined_mask = valid_positions & flat_mask
            
            if combined_mask.sum() == 0:
                return None
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            flat_labels_valid = flat_labels.clone()
            flat_labels_valid[~valid_positions] = 0  # 临时替换，不影响计算
            
            flat_loss = loss_fct(flat_logits, flat_labels_valid)
            masked_loss = flat_loss * combined_mask.float()
            
            translation_loss = masked_loss.sum() / combined_mask.sum()
            
            return translation_loss
            
        except Exception as e:
            print(f"Warning: Failed to compute translation loss: {str(e)}")
            return None
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        重写log方法，将translation_loss添加到常规日志中。
        """
        # 如果有存储的translation_loss，添加到logs中
        if self._current_translation_loss is not None:
            logs["translation_loss"] = self._current_translation_loss
            # 如果logs中有loss，计算ratio
            if "loss" in logs:
                logs["loss_ratio"] = self._current_translation_loss / logs["loss"] if logs["loss"] > 0 else 0.0
            # 清空临时存储
            self._current_translation_loss = None
        
        # 调用父类的log方法
        super().log(logs, start_time)
    
    def _create_translation_mask_v2(self, input_ids: torch.Tensor, labels: torch.Tensor) -> Optional[torch.Tensor]:
        """
        改进版mask创建：使用tokenizer解码来精确定位翻译起始位置。
        """
        try:
            batch_size, seq_length = labels.shape
            translation_mask = torch.zeros_like(labels, dtype=torch.bool)
            
            if self._processing_class is None:
                # Fallback到启发式方法
                return self._create_translation_mask_heuristic(labels)
            
            for batch_idx in range(batch_size):
                sample_labels = labels[batch_idx]
                
                # 找到有效的label位置（非-100）
                valid_mask = sample_labels != -100
                valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                
                if len(valid_indices) == 0:
                    continue
                
                # 解码有效的labels来查找翻译起始位置
                valid_labels = sample_labels[valid_mask]
                
                try:
                    decoded_text = self._processing_class.decode(valid_labels, skip_special_tokens=False)
                    
                    # 查找翻译标记
                    translation_patterns = [
                        r"The translation is:\s*",
                        r"So the translation is:\s*",
                        r"translation is:\s*",
                    ]
                    
                    translation_start_pos = None
                    for pattern in translation_patterns:
                        match = re.search(pattern, decoded_text, re.IGNORECASE)
                        if match:
                            translation_start_pos = match.end()
                            break
                    
                    if translation_start_pos is not None:
                        # 找到翻译起始位置对应的token索引
                        # 这里需要将字符位置映射回token位置
                        # 简化处理：从匹配位置之后开始标记所有剩余token
                        
                        # 重新编码前半部分来确定token数量
                        before_translation = decoded_text[:translation_start_pos]
                        before_tokens = self._processing_class.encode(before_translation, add_special_tokens=False)
                        
                        # 从这个token位置开始标记
                        start_token_idx = len(before_tokens)
                        
                        # 标记从start_token_idx开始的所有有效token
                        for i, valid_idx in enumerate(valid_indices):
                            if i >= start_token_idx:
                                translation_mask[batch_idx, valid_idx] = True
                    else:
                        # 如果找不到标记，使用启发式：标记最后30%
                        num_valid = len(valid_indices)
                        start_idx = int(num_valid * 0.7)
                        for idx in valid_indices[start_idx:]:
                            translation_mask[batch_idx, idx] = True
                            
                except Exception as decode_error:
                    # 解码失败，使用启发式方法
                    num_valid = len(valid_indices)
                    start_idx = int(num_valid * 0.7)
                    for idx in valid_indices[start_idx:]:
                        translation_mask[batch_idx, idx] = True
            
            return translation_mask
            
        except Exception as e:
            print(f"Warning: Failed to create translation mask v2: {str(e)}")
            return self._create_translation_mask_heuristic(labels)
    
    def _create_translation_mask_heuristic(self, labels: torch.Tensor) -> torch.Tensor:
        """
        启发式方法：标记最后30%的有效token为翻译部分。
        """
        batch_size, seq_length = labels.shape
        translation_mask = torch.zeros_like(labels, dtype=torch.bool)
        
        for batch_idx in range(batch_size):
            valid_indices = (labels[batch_idx] != -100).nonzero(as_tuple=True)[0]
            if len(valid_indices) > 0:
                num_valid = len(valid_indices)
                start_idx = int(num_valid * 0.7)
                for idx in valid_indices[start_idx:]:
                    translation_mask[batch_idx, idx] = True
        
        return translation_mask
