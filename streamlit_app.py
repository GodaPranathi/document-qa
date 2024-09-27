import streamlit as st
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import re

def highlight_text(text, term):
    highlighted_text = re.sub(f"({term})", r'<mark>\1</mark>', text, flags=re.IGNORECASE)
    return highlighted_text

@st.cache_resource
def load_models():
    RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16).eval()
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    
    return model, processor, RAG

if 'is_indexed' not in st.session_state:
    st.session_state['is_indexed'] = False

st.title("Image to Text Extraction and Search with Highlighting")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Save the uploaded image to a temporary file
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(uploaded_file)
    images = [image]
    st.image(image, caption='Uploaded Image', use_column_width=True)

    model, processor, RAG = load_models()

    # Text Extraction from Image
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
    inputs = inputs.to("cpu")


    # Generate the text from the image using the model
    generated_ids = model.generate(**inputs, max_new_tokens=5000)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    extracted_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    extracted_text = "\n".join(extracted_text)  # Convert list to a single string
    
    st.subheader("Extracted Text:")
    st.write(extracted_text)

    # Save the extracted text to a file
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)

    #  Search Query
    query = st.text_input("Search in Extracted Text", "")
    
    if query:
        # If the query is a single word, highlight its occurrences
        if len(query.split()) == 1:
            # Highlight the search term in the extracted text
            highlighted_text = highlight_text(extracted_text, query)
            st.subheader("Search Result (Word Occurrences):")
            st.markdown(highlighted_text, unsafe_allow_html=True)
        
        # If the query is more than one word, use RAG for Intelli search
        else:
            # Only index the image once
            if not st.session_state['is_indexed']:
                try:
                    RAG.index(
                        input_path=temp_file_path,  # Use the local file path for indexing
                        index_name="image_index",   # index will be saved at index_root/index_name/
                        store_collection_with_index=False,
                        overwrite=True
                    )
                    st.session_state['is_indexed'] = True  # Mark document as indexed
                except Exception as e:
                    st.error(f"Error during indexing: {str(e)}")
            
            # Perform search using the query
            try:
                results = RAG.search(query, k=1)
                query_image_index = results[0]["page_num"] - 1
                
                # Get the result text related to the query
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
                
                # Generate the answer using the RAG model
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
                inputs = inputs.to("cpu")
                
                generated_ids_query = model.generate(**inputs, max_new_tokens=1000)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids_query)
                ]
                query_result = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                
                # Highlight the query within the result
                highlighted_result = highlight_text("\n".join(query_result), query)
                
                # Display the query result
                st.subheader("Search Result (Intelli Answer):")
                st.markdown(highlighted_result, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error during search: {str(e)}")
