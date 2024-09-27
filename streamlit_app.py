import streamlit as st
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import re
import os

# Function to highlight search terms in the text
def highlight_text(text, term):
    highlighted_text = re.sub(f"({term})", r'<mark>\1</mark>', text, flags=re.IGNORECASE)
    return highlighted_text

# Load models only once when the app starts
@st.cache_resource
def load_models():
    # Load RAG model for multimodal interaction
    RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")
    
    # Load Qwen2-VL model for text generation from images
    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct",
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16).cuda().eval()
    
    # Load the processor for Qwen2-VL model
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    
    return model, processor, RAG

# Streamlit interface
st.title("Image to Text Extraction and Search with Highlighting")

# Image upload widget
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Save the uploaded image to a temporary file
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    image = Image.open(uploaded_file)
    images=[image]
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Load models
    model, processor, RAG = load_models()
    
    # Step 1: Text Extraction from Image
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": "Extract the text from this image."},
            ],
        }
    ]

    # Process the image and text for input
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Generate the text from the image using the model
    generated_ids = model.generate(**inputs, max_new_tokens=5000)
    
    # Trim generated text to exclude initial tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    extracted_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Display the extracted text
    st.subheader("Extracted Text:")
    st.write("\n".join(extracted_text))
    
    # Save the extracted text
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(extracted_text))

    # Step 2: Search Query
    query = st.text_input("Search in Extracted Text", "")
    
    if query:
        # Intelli Search using RAG
        RAG.index(
            input_path=temp_file_path,  # Use the local file path for indexing
            index_name="image_index",  # index will be saved at index_root/index_name/
            store_collection_with_index=False,
            overwrite=True
        )
        
        # Perform search using the query
        results = RAG.search(query, k=1)
        query_image_index = results[0]["page_num"] - 1
        query_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": images[query_image_index],
                    },
                    {"type": "text", "text": query},
                ],
            }
        ]
        
        text = processor.apply_chat_template(
            query_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        generated_ids_query = model.generate(**inputs, max_new_tokens=1000)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids_query)
        ]
        query_result = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Highlight the search term in the extracted text
        highlighted_text = highlight_text("\n".join(extracted_text), query)
        
        # Display the query result with highlighted terms
        st.subheader("Search Result:")
        st.markdown(highlighted_text, unsafe_allow_html=True)
    
    # Clean up the temporary file after processing
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
