from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import random
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import BytesIO
from torch.nn.functional import cosine_similarity
import pandas as pd
from ultils import json_to_dataframe
from similarity import compute_cosine_similarity, contains_word, extract_words

model = SentenceTransformer('paraphrase-mpnet-base-v2')
img_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

app = Flask(__name__)
CORS(app)

@app.route('/prerecommend', methods=['POST'])
def pre_recommend():
    data = request.json
    products = data.get("products")
    number_of_items = data.get("numberOfItems", 4) 

    df = json_to_dataframe(products)
    df_types = df.groupby("product_type").apply(
        lambda group: [
            (row["name"], row["shopify_id"])
            for _, row in group.iterrows()
        ]
    )

    texts=df_types.keys()
    embeddings = {}
    for text in texts:
        text = str(text)
        embedding = model.encode(text)
        embeddings[text] = embedding

    df_new = {}
    for text, vector in embeddings.items():
        other_texts = {k: v for k, v in embeddings.items() if k != text}
        similarities = compute_cosine_similarity(vector, list(other_texts.values()))

        adjusted_similarities = []
        words_in_text = extract_words(text)

        for i, other_text in enumerate(other_texts.keys()):
            similarity = similarities[i]
            words_in_other_text = extract_words(other_text)

            filtered_words_in_text = words_in_text - {"men", "women", "s"}
            filtered_words_in_other_text = words_in_other_text - {"men", "women", "s"}

            common_words = filtered_words_in_text & filtered_words_in_other_text

            if common_words:
                similarity -= 0.5  

            adjusted_similarities.append(similarity)

        sorted_texts = [t for t, _ in sorted(zip(other_texts.keys(), adjusted_similarities), key=lambda x: x[1], reverse=True)]
        df_new[text] = sorted_texts

    recommend_items = {}
    for item in df.itertuples():
        type = item.product_type
        id = item.shopify_id
        i=0
        recommend_items[id]=[]

        recommend_types = df_new[type]
        for recommend_type in recommend_types:
            chosen_product_index = random.randint(0, len(df_types[recommend_type])-1)
            chosen_product_id = df_types[recommend_type][chosen_product_index][1]
            if len(chosen_product_id)>-1:
                recommend_items[id].append(chosen_product_id)
                i+=1
            if i==number_of_items:
                break
    for key, value in recommend_items.items():
        print(key, value)

    return jsonify(recommend_items)
  

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    products = data.get("products")
    number_of_items = data.get("numberOfItems", 4) 

    df = json_to_dataframe(products)
    df["name_embedding"] = None
    df["image_embedding"] = None
    df["num_reviews"] = None
    results = []

    for row in df.itertuples():
        try:
            name_embedding = model.encode(row.name)
            
            response = requests.get(row.image)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                image_embedding = img_model.get_image_features(**inputs)
                image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
                image_embedding = image_embedding.reshape((512,))

            results.append({
                "index": row.Index,
                "name_embedding": name_embedding,
                "image_embedding": image_embedding,
                "num_review": 0
            })
        except Exception as e:
            print(f"Lỗi với sản phẩm {row.name}: {e}")

    for result in results:
        df.at[result["index"], "name_embedding"] = result["name_embedding"]
        df.at[result["index"], "image_embedding"] = result["image_embedding"]
        df.at[result["index"], "num_review"] = result["num_review"]

    df_types = df.groupby("product_type").apply(
        lambda group: [
            (row["shopify_id"], row["name"], row["name_embedding"], row["image_embedding"], row["num_review"])
            for _, row in group.iterrows()
        ]
    )

    texts=df_types.keys()
    embeddings = {}
    for text in texts:
        text = str(text)
        embedding = model.encode(text)
        embeddings[text] = embedding

    df_new = {}
    for text, vector in embeddings.items():
        other_texts = {k: v for k, v in embeddings.items() if k != text}
        similarities = compute_cosine_similarity(vector, list(other_texts.values()))

        adjusted_similarities = []
        words_in_text = extract_words(text)

        for i, other_text in enumerate(other_texts.keys()):
            similarity = similarities[i]
            words_in_other_text = extract_words(other_text)

            filtered_words_in_text = words_in_text - {"men", "women", "s"}
            filtered_words_in_other_text = words_in_other_text - {"men", "women", "s"}

            common_words = filtered_words_in_text & filtered_words_in_other_text

            if common_words:
                similarity -= 0.5  

            adjusted_similarities.append(similarity)

        sorted_texts = [t for t, _ in sorted(zip(other_texts.keys(), adjusted_similarities), key=lambda x: x[1], reverse=True)]
        df_new[text] = sorted_texts

    recommend_items = {}
    for item in df.itertuples():
        type = item.product_type
        id = item.shopify_id
        name_embedding = item.name_embedding
        image_embedding = item.image_embedding
        if type not in df_new or image_embedding==None:
            continue

        recommend_types = df_new[type]
        recommend_items[id] = []
        i=0
        for recommend_type in recommend_types:
            max_similarity = -1
            chosen_product_id = ""
            for product_id, product_name, embedding_product_name, embedding_img, num_reviews in df_types[recommend_type]:
                try:
                    similarity1 = cosine_similarity([name_embedding ], [embedding_product_name  ])[0][0]
                    similarity2 = cosine_similarity([ image_embedding.cpu().numpy().reshape((512,))], [embedding_img.cpu().numpy().reshape((512,))])[0][0]
                    similarity = 0.5*similarity1 + 0.5*similarity2
                    similarity = similarity*(1-0.1*num_reviews)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        chosen_product_id = product_id
                except Exception as e:
                    print(f"Lỗi với sản phẩm {product_name}: {e}")

            if len(chosen_product_id)>0:
                recommend_items[id].append(chosen_product_id)
                df.loc[df['name'] == chosen_product_id, 'num_review'] += 1
                i+=1
            if i==number_of_items:
                break
    
    return jsonify(recommend_items)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
