import re
from sklearn.metrics.pairwise import cosine_similarity

# Hàm kiểm tra từ khóa độc lập
def contains_word(word, text):
    pattern = rf'\b{word}\b'  # Tìm từ chính xác, không phải là một phần của từ khác
    return re.search(pattern, text.lower()) is not None

# Hàm tách các từ từ một chuỗi
def extract_words(text):
    return set(re.findall(r'\b\w+\b', text.lower()))  # Lấy các từ độc lập

# Hàm tính độ tương đồng cosine
def compute_cosine_similarity(vector, other_vectors):
    return cosine_similarity([vector], other_vectors)[0]