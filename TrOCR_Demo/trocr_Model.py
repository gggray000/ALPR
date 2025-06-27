from transformers import ViTModel, RobertaConfig, RobertaForCausalLM, RobertaTokenizer, TrOCRProcessor, ViTImageProcessor, VisionEncoderDecoderModel

encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
decoder_config = RobertaConfig.from_pretrained('roberta-base')
decoder_config.is_decoder = True
decoder_config.add_cross_attention = True
decoder = RobertaForCausalLM(config=decoder_config)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.sep_token_id

model.save_pretrained("models/new-trocr-model")
processor.save_pretrained("models/new-trocr-model")