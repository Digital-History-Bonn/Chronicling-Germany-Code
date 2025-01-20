"""OCR Prediction"""

import argparse
import glob
import json
import os
from typing import Tuple, List

import torch
import torch.nn.functional as F
from skimage import io

from bs4 import BeautifulSoup, ResultSet
from tqdm import tqdm, trange

from src.cgprocess.OCR.Transformer import ALPHABET, PAD_HEIGHT, PAD_WIDTH
from src.cgprocess.OCR.Transformer.ocr_engine import transformer
from src.cgprocess.OCR.Transformer.ocr_engine.transformer import TransformerOCR
from src.cgprocess.OCR.shared.tokenizer import Tokenizer
from src.cgprocess.OCR.shared.utils import get_bbox
from src.cgprocess.layout_segmentation.processing.read_xml import xml_polygon_to_polygon_list


def read_xml(soup: BeautifulSoup) -> Tuple[ResultSet, List[torch.Tensor]]:
    """
    Reads out textlines and corresponding bounding boxes form soup.
    Args:
        soup: soup object with textline annotations.

    Returns:
        text_lines: ResultSet with textlines
        bboxes: Bounding boxes from textlines.
    """
    page = soup.find('Page')
    bboxes = []

    text_lines = page.find_all('TextLine')
    for line in text_lines:
        region_polygon = torch.tensor(xml_polygon_to_polygon_list(line.Coords["points"]))
        bboxes.append(torch.tensor(get_bbox(region_polygon)))

    return text_lines, bboxes


def replace_text(soup: BeautifulSoup, new_text: str) -> None:
    """
    Replaces text form xml file with predicted text.
    Args:
        soup: Soup object with textline annotations.
        new_text: predicted text.
    """
    # Find the first occurrence of the tag
    tag = soup.find('Unicode')

    if tag and tag.text:
        # Replace the text inside the tag
        tag.string.replace_with(new_text)


def predict(model: transformer.TransformerOCR,
            tokenizer: Tokenizer,
            images: torch.Tensor,
            max_length: int = 100,
            start_token_idx: int = 1,
            eos_token_idx: int = 3,
            nan_token_idx: int= 2,
            predict_nan: bool = False) -> str:
    """
    Perform autoregressive prediction using the transformer decoder.

    Args:
        model (transformer.TransformerOCR): The transformer to predict the text.
        tokenizer (Tokenizer): The tokenizer to use for reverse ids.
        images: Input tensor for the encoder, expected shape [batch_size, seq_length].
        max_length: Maximum length of sequence for prediction
        gth of the sequence to be generated.
        start_token_idx: Index of the start token in the alphabet.
        eos_token_idx: Index of the end token in the alphabet.
        nan_token_idx: Index of the nan token in the alphabet.
        predict_nan: Whether to predict nan or not.

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
    for _ in range(max_length - 1):  # Already have <START> as the first token
        # Get the current length of the generated sequence
        tgt_len = generated_sequences.size(0)

        # Step 3a: Get the mask for the current length of the generated sequence
        dec_mask = model.get_mask(tgt_len).to(images.device)

        # Step 3b: Get the embeddings and apply positional encoding
        tgt_embs = model.dec_embeder(generated_sequences)
        tgt_embs = model.pos_encoder(tgt_embs)

        # Step 3c: Pass through the decoder
        decoder_output = model.trans_decoder(tgt_embs, encoder_output, tgt_mask=dec_mask)

        # Step 3d: Project the decoder output to vocabulary size and get the next token
        # Use the last step's output for next token
        output_logits = model.dec_out_proj(decoder_output[-1, :, :])
        if not predict_nan:
            output_logits[:, nan_token_idx] = -float('inf')

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


def predict_and_write(model: TransformerOCR,
                      tokenizer: Tokenizer,
                      device: torch.device,
                      image_path: str,
                      anno_path: str,
                      out_path: str) -> None:
    """
    Predicts textlines based on annotated lines and writes them into the xml file.

    Args:
        model: model to use for prediction
        tokenizer: tokenizer to use for prediction
        device: device to use for prediction
        image_path: path to image
        anno_path: path to annotation
        out_path: path to save the resulting xml file
    """
    if os.path.exists(out_path):
        return

    image = torch.tensor(io.imread(image_path)).permute(2, 0, 1).to(device)

    with open(anno_path, "r", encoding="utf-8") as file:
        data = file.read()

    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    textlines, bboxes = read_xml(soup)

    for line, bbox in tqdm(zip(textlines, bboxes), total=len(textlines), desc="precessing lines",
                           disable=True):
        # pylint: disable=duplicate-code
        crop = image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]

        pad_height = max(0, PAD_HEIGHT - crop.shape[1])
        pad_width = max(0, PAD_WIDTH - crop.shape[2])
        crop = F.pad(crop, (pad_width, 0, pad_height, 0), "constant", 0)
        crop = crop[:, :PAD_HEIGHT]

        text = predict(model, tokenizer, crop[None].float() / 255)

        replace_text(line, text)

    # save results
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as file:
        file.write(soup.prettify()
                   .replace("<Unicode>\n      ", "<Unicode>")
                   .replace("\n     </Unicode>", "</Unicode>"))


def get_args() -> argparse.Namespace:
    """
    Defines arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="train Pero OCR")

    parser.add_argument(
        "--model",
        "-m",
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

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="path to output folder for predicted xml files."
    )

    return parser.parse_args()


def main() -> None:
    """Predicts OCR and write it into xml file using textline annotations."""
    args = get_args()

    print(f"{torch.cuda.is_available()=}")
    device = (
        torch.device(f"cuda:{0}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"using {device}")

    with open("src/OCR/Transformer/config.json", "r", encoding='utf-8') as file:
        json_data = json.load(file)

    model: transformer.TransformerOCR = transformer.build_net(net=json_data,
                                                              input_height=PAD_HEIGHT,
                                                              input_channels=3,
                                                              nb_output_symbols=len(ALPHABET) - 2)
    model.load_state_dict(torch.load(f"models/{args.model}.pt"))
    model.to(device)

    tokenizer = Tokenizer(ALPHABET)

    images = glob.glob(f"{args.data}/*.jpg")
    annos = [f"{x[:-4]}.xml" for x in images]
    outputs = [f"{args.output}/{os.path.basename(x)[:-4]}.xml" for x in images]

    for i in trange(len(images), desc="preprocessing images"):
        predict_and_write(model,
                          tokenizer,
                          device,
                          image_path=images[i],
                          anno_path=annos[i],
                          out_path=outputs[i])


if __name__ == '__main__':
    main()
