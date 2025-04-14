from typing import List, Optional, Tuple, Union
import torch
from transformers import Qwen2VLForConditionalGeneration
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
from torch.nn import CrossEntropyLoss, LayerNorm

class Qwen2VLGRPO(Qwen2VLForConditionalGeneration):
    def __init__(self, config):
       super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Optional[int] = None
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:, -logits_to_keep:, :])

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

# class Qwen2VLForConditionalGeneration(Qwen2VLPreTrainedModel, GenerationMixin):
    # _tied_weights_keys = ["lm_head.weight"]

    # def __init__(self, config):
    #     super().__init__(config)
    #     self.visual = Qwen2VisionTransformerPretrainedModel._from_config(config.vision_config)
    #     self.model = Qwen2VLModel(config)
    #     self.vocab_size = config.vocab_size
    #     self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    #     self.rope_deltas = None  # cache rope_deltas here

    #     # Initialize weights and apply final processing
    #     self.post_init()

    # def get_input_embeddings(self):
    #     return self.model.embed_tokens

    # def set_input_embeddings(self, value):
    #     self.model.embed_tokens = value

    # def get_output_embeddings(self):
    #     return self.lm_head

    # def set_output_embeddings(self, new_embeddings):
    #     self.lm_head = new_embeddings

    # def set_decoder(self, decoder):
    #     self.model = decoder

    # def get_decoder(self):
    #     return self.model

    # def get_rope_index(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     image_grid_thw: Optional[torch.LongTensor] = None,
    #     video_grid_thw: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    #     Explanation:
    #         Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

    #         For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
    #         Examples:
    #             input_ids: [T T T T T], here T is for text.
    #             temporal position_ids: [0, 1, 2, 3, 4]
    #             height position_ids: [0, 1, 2, 3, 4]
    #             width position_ids: [0, 1, 2, 3, 4]

    #         For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
    #         and 1D rotary position embeddin for text part.
    #         Examples:
    #             Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
    #             input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
    #             vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    #             vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    #             vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    #             text temporal position_ids: [3, 4, 5, 6, 7]
    #             text height position_ids: [3, 4, 5, 6, 7]
    #             text width position_ids: [3, 4, 5, 6, 7]
    #             Here we calculate the text start position_ids as the max vision position_ids plus 1.

    #     Args:
    #         input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
    #             Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
    #             it.
    #         image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
    #             The temporal, height and width of feature shape of each image in LLM.
    #         video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
    #             The temporal, height and width of feature shape of each video in LLM.
    #         attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
    #             Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

    #             - 1 for tokens that are **not masked**,
    #             - 0 for tokens that are **masked**.

    #     Returns:
    #         position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
    #         mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    #     """
    #     spatial_merge_size = self.config.vision_config.spatial_merge_size
    #     image_token_id = self.config.image_token_id
    #     video_token_id = self.config.video_token_id
    #     vision_start_token_id = self.config.vision_start_token_id
    #     mrope_position_deltas = []
    #     if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
    #         total_input_ids = input_ids
    #         if attention_mask is None:
    #             attention_mask = torch.ones_like(total_input_ids)
    #         position_ids = torch.ones(
    #             3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
    #         )
    #         image_index, video_index = 0, 0
    #         for i, input_ids in enumerate(total_input_ids):
    #             input_ids = input_ids[attention_mask[i] == 1]
    #             image_nums, video_nums = 0, 0
    #             vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
    #             vision_tokens = input_ids[vision_start_indices + 1]
    #             image_nums = (vision_tokens == image_token_id).sum()
    #             video_nums = (vision_tokens == video_token_id).sum()
    #             input_tokens = input_ids.tolist()
    #             llm_pos_ids_list: list = []
    #             st = 0
    #             remain_images, remain_videos = image_nums, video_nums
    #             for _ in range(image_nums + video_nums):
    #                 if image_token_id in input_tokens and remain_images > 0:
    #                     ed_image = input_tokens.index(image_token_id, st)
    #                 else:
    #                     ed_image = len(input_tokens) + 1
    #                 if video_token_id in input_tokens and remain_videos > 0:
    #                     ed_video = input_tokens.index(video_token_id, st)
    #                 else:
    #                     ed_video = len(input_tokens) + 1
    #                 if ed_image < ed_video:
    #                     t, h, w = (
    #                         image_grid_thw[image_index][0],
    #                         image_grid_thw[image_index][1],
    #                         image_grid_thw[image_index][2],
    #                     )
    #                     image_index += 1
    #                     remain_images -= 1
    #                     ed = ed_image
    #                 else:
    #                     t, h, w = (
    #                         video_grid_thw[video_index][0],
    #                         video_grid_thw[video_index][1],
    #                         video_grid_thw[video_index][2],
    #                     )
    #                     video_index += 1
    #                     remain_videos -= 1
    #                     ed = ed_video
    #                 llm_grid_t, llm_grid_h, llm_grid_w = (
    #                     t.item(),
    #                     h.item() // spatial_merge_size,
    #                     w.item() // spatial_merge_size,
    #                 )
    #                 text_len = ed - st

    #                 st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
    #                 llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

    #                 t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
    #                 h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
    #                 w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
    #                 llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
    #                 st = ed + llm_grid_t * llm_grid_h * llm_grid_w

    #             if st < len(input_tokens):
    #                 st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
    #                 text_len = len(input_tokens) - st
    #                 llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

    #             llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    #             position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
    #             mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
    #         mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    #         return position_ids, mrope_position_deltas
    #     else:
    #         if attention_mask is not None:
    #             position_ids = attention_mask.long().cumsum(-1) - 1
    #             position_ids.masked_fill_(attention_mask == 0, 1)
    #             position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
    #             max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
    #             mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    #         else:
    #             position_ids = (
    #                 torch.arange(input_ids.shape[1], device=input_ids.device)
    #                 .view(1, 1, -1)
    #                 .expand(3, input_ids.shape[0], -1)
    #             )
    #             mrope_position_deltas = torch.zeros(
    #                 [input_ids.shape[0], 1],
    #                 device=input_ids.device,
    #                 dtype=input_ids.dtype,
    #             )

    #         return position_ids, mrope_position_deltas

    # @add_start_docstrings_to_model_forward(QWEN2_VL_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Qwen2VLCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    # def forward(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    #     pixel_values: Optional[torch.Tensor] = None,
    #     pixel_values_videos: Optional[torch.FloatTensor] = None,
    #     image_grid_thw: Optional[torch.LongTensor] = None,
    #     video_grid_thw: Optional[torch.LongTensor] = None,
    #     rope_deltas: Optional[torch.LongTensor] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     logits_to_keep: Optional[int] = None
    # ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
    #     r"""
    #     Args:
    #         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
    #             Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
    #             config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
    #             (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    #     Returns:

    #     Example:

    #     ```python
    #     >>> from PIL import Image
    #     >>> import requests
    #     >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    #     >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    #     >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    #     >>> messages = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "image"},
    #                 {"type": "text", "text": "What is shown in this image?"},
    #             ],
    #         },
    #     ]
    #     >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    #     >>> image = Image.open(requests.get(url, stream=True).raw)

    #     >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #     >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

    #     >>> # Generate
    #     >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    #     >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    #     "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
    #     ```"""

    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    #     if inputs_embeds is None:
    #         inputs_embeds = self.model.embed_tokens(input_ids)
    #         if pixel_values is not None:
    #             pixel_values = pixel_values.type(self.visual.get_dtype())
    #             image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
    #             n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
    #             n_image_features = image_embeds.shape[0]
    #             if n_image_tokens != n_image_features:
    #                 raise ValueError(
    #                     f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
    #                 )
    #             image_mask = (
    #                 (input_ids == self.config.image_token_id)
    #                 .unsqueeze(-1)
    #                 .expand_as(inputs_embeds)
    #                 .to(inputs_embeds.device)
    #             )
    #             image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    #             inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    #         if pixel_values_videos is not None:
    #             pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
    #             video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
    #             n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
    #             n_video_features = video_embeds.shape[0]
    #             if n_video_tokens != n_video_features:
    #                 raise ValueError(
    #                     f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
    #                 )
    #             video_mask = (
    #                 (input_ids == self.config.video_token_id)
    #                 .unsqueeze(-1)
    #                 .expand_as(inputs_embeds)
    #                 .to(inputs_embeds.device)
    #             )
    #             video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    #             inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    #         if attention_mask is not None:
    #             attention_mask = attention_mask.to(inputs_embeds.device)

    #     # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    #     if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
    #         # calculate RoPE index once per generation in the pre-fill stage only
    #         if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
    #             position_ids, rope_deltas = self.get_rope_index(
    #                 input_ids, image_grid_thw, video_grid_thw, attention_mask
    #             )
    #             self.rope_deltas = rope_deltas
    #         # then use the prev pre-calculated rope-deltas to get the correct position ids
    #         else:
    #             batch_size, seq_length, _ = inputs_embeds.shape
    #             delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
    #             position_ids = torch.arange(seq_length, device=inputs_embeds.device)
    #             position_ids = position_ids.view(1, -1).expand(batch_size, -1)
    #             if cache_position is not None:  # otherwise `deltas` is an int `0`
    #                 delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
    #             position_ids = position_ids.add(delta)
    #             position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    #     outputs = self.model(
    #         input_ids=None,
    #         position_ids=position_ids,
    #         attention_mask=attention_mask,
    #         past_key_values=past_key_values,
    #         inputs_embeds=inputs_embeds,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #         cache_position=cache_position,
    #     )

    #     hidden_states = outputs[0]
    #     logits = self.lm_head(hidden_states[:, -logits_to_keep:, :])

    #     loss = None
    #     if labels is not None:
    #         # Upcast to float if we need to compute the loss to avoid potential precision issues
    #         logits = logits.float()
    #         # Shift so that tokens < n predict n
    #         shift_logits = logits[..., :-1, :].contiguous()
    #         shift_labels = labels[..., 1:].contiguous()
    #         # Flatten the tokens
    #         loss_fct = CrossEntropyLoss()
    #         shift_logits = shift_logits.view(-1, self.config.vocab_size)
    #         shift_labels = shift_labels.view(-1)
    #         # Enable model parallelism
    #         shift_labels = shift_labels.to(shift_logits.device)
    #         loss = loss_fct(shift_logits, shift_labels)

    #     if not return_dict:
    #         output = (logits,) + outputs[1:]
    #         return (loss,) + output if loss is not None else output

    #     return Qwen2VLCausalLMOutputWithPast(
    #         loss=loss,
    #         logits=logits,
    #         past_key_values=outputs.past_key_values,
    #         hidden_states=outputs.hidden_states,
    #         attentions=outputs.attentions,
    #         rope_deltas=self.rope_deltas,
    #     )

    # def prepare_inputs_for_generation(
    #     self,
    #     input_ids,
    #     past_key_values=None,
    #     attention_mask=None,
    #     inputs_embeds=None,
    #     cache_position=None,
    #     position_ids=None,
    #     use_cache=True,
    #     pixel_values=None,
    #     pixel_values_videos=None,
    #     image_grid_thw=None,
    #     video_grid_thw=None,
    #     **kwargs,
    # ):
    #     # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

    #     # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
    #     # Exception 1: when passing input_embeds, input_ids may be missing entries
    #     # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
    #     if past_key_values is not None:
    #         if inputs_embeds is not None:  # Exception 1
    #             input_ids = input_ids[:, -cache_position.shape[0] :]
    #         elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
    #             input_ids = input_ids[:, cache_position]

    #     if cache_position[0] != 0:
    #         pixel_values = None
    #         pixel_values_videos = None

    #     # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    #     if inputs_embeds is not None and cache_position[0] == 0:
    #         model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
    #     else:
    #         model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

    #     if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
    #         if model_inputs["inputs_embeds"] is not None:
    #             batch_size, sequence_length, _ = inputs_embeds.shape
    #             device = inputs_embeds.device
    #         else:
    #             batch_size, sequence_length = input_ids.shape
    #             device = input_ids.device

    #         attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
    #             attention_mask,
    #             sequence_length=sequence_length,
    #             target_length=past_key_values.get_max_cache_shape(),
    #             dtype=self.lm_head.weight.dtype,
    #             device=device,
    #             cache_position=cache_position,
    #             batch_size=batch_size,
    #             config=self.config,
    #             past_key_values=past_key_values,
    #         )

    #     model_inputs.update(
    #         {
    #             "position_ids": position_ids,
    #             "past_key_values": past_key_values,
    #             "use_cache": use_cache,
    #             "attention_mask": attention_mask,
    #             "pixel_values": pixel_values,
    #             "pixel_values_videos": pixel_values_videos,
    #             "image_grid_thw": image_grid_thw,
    #             "video_grid_thw": video_grid_thw,
    #             "cache_position": cache_position,
    #         }
    #     )
    #     return model_inputs