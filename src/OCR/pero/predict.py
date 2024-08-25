import argparse
import json

import torch

from src.OCR.pero.dataset import Dataset
from src.OCR.pero.ocr_engine import transformer
from src.OCR.pero.trainer import ALPHABET, CROP_HEIGHT
from src.OCR.pero.utils import Tokenizer


def predict(model: transformer.TransformerOCR,
            tokenizer: Tokenizer,
            images: torch.Tensor,
            max_length: int = 100,
            start_token_idx: int = 1,
            eos_token_idx: int = 3):
    """
    Perform autoregressive prediction using the transformer decoder.

    Args:
        model (transformer.TransformerOCR): The transformer to predict the text.
        tokenizer (Tokenizer): The tokenizer to use for reverse ids.
        images: Input tensor for the encoder, expected shape [batch_size, seq_length].
        max_length: Maximum len
        gth of the sequence to be generated.
        start_token_idx: Index of the start token in the vocabulary.
        eos_token_idx: Index of the end token in the vocabulary.

    Returns:
        generated_sequences: The predicted sequences, shape [batch_size, max_length].
    """

    # Step 1: Encode the input sequence
    encoder_output = model.encode(images)

    # Step 2: Initialize the generated sequences with the start token
    batch_size = images.size(0)
    generated_sequences = torch.full((1, batch_size), start_token_idx, dtype=torch.long).to(
        images.device)

    # Step 3: Iteratively generate the sequence
    for i in range(max_length - 1):  # Already have <START> as the first token
        # Get the current length of the generated sequence
        tgt_len = generated_sequences.size(0)

        # Step 3a: Get the mask for the current length of the generated sequence
        dec_mask = model.get_mask(tgt_len).to(images.device)

        # Step 3b: Get the embeddings and apply positional encoding
        tgt_embs = model.dec_embeder(generated_sequences)
        tgt_embs = model.pos_encoder(tgt_embs)

        # print(f"{i}: {tgt_embs.shape=}")

        # Step 3c: Pass through the decoder
        decoder_output = model.trans_decoder(tgt_embs, encoder_output, tgt_mask=dec_mask)
        # print(f"{i}: {decoder_output.shape=}")

        # Step 3d: Project the decoder output to vocabulary size and get the next token
        # Use the last step's output for next token
        output_logits = model.dec_out_proj(decoder_output[-1, :, :])
        next_token = output_logits.argmax(dim=-1,
                                          keepdim=True)  # Get the most likely next token

        # Append the next token to the generated sequence
        generated_sequences = torch.cat([generated_sequences, next_token], dim=0)

        # Stop if all sequences have generated the <eos> token
        if (next_token == eos_token_idx).all():
            break

    # Return batch-first output shape [batch_size, seq_length]
    pred = generated_sequences.permute(1, 0)
    return tokenizer.to_text(pred[0])


def get_args() -> argparse.Namespace:
    """
    Defines arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="train Pero OCR")

    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default="model",
        help="Name of the model and the log files."
    )

    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default=None,
        help="path for folder with images jpg files to predict."
    )

    return parser.parse_args()


def main():
    args = get_args()
    print(f"{args=}")

    print(f"{torch.cuda.is_available()=}")
    device = (
        torch.device(f"cuda:{0}")
        if torch.cuda.is_available() and 0 >= 0
        else torch.device("cpu")
    )
    print(f"using {device}")

    with open("src/OCR/pero/config.json", "r") as file:
        json_data = json.load(file)

    model: transformer.TransformerOCR = transformer.build_net(net=json_data,
                                                              input_height=CROP_HEIGHT,
                                                              input_channels=3,
                                                              nb_output_symbols=len(ALPHABET) - 2)
    model.load_state_dict(torch.load(f"models/{args.name}.pt"))
    model.to(device)

    tokenizer = Tokenizer(ALPHABET)

    validset = Dataset(image_path=args.data,
                       target_path=args.data,
                       alphabet=ALPHABET,
                       pad=False,
                       cache_images=True)

    for i in range(100):
        image, target, text = validset[i]

        pred_text = predict(model, tokenizer, image[None].to(device))

        print()
        print(f"{text=}")
        print(f"{pred_text=}")
        print()


if __name__ == '__main__':
    main()
