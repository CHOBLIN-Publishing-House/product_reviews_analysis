import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sklearn
import pandas as pd


st.set_page_config(layout="centered")  # Это возвращает боковые отступы



# Загружаем модель
model = joblib.load('logistic_regression_sentiment.pkl')

# Загружаем векторизатор
vectorizer = joblib.load("tfidf_vectorizer.pkl")



st.header('Определение тональности отзыва')
st.write('Точность прогноза 87%')

lemmatizer = WordNetLemmatizer()

# Загружаем NLTK данные (если их нет)
nltk.download('stopwords')
english_stopwords = set(stopwords.words('english'))

def normalize_text(text):
  # Приводим к нижнему регистру
  text = text.lower()

  # Удаляем пунктуацию и не-буквы
  text = re.sub(r'[^a-z\s]', ' ', text)

  # Разбиваем на слова
  words = text.split()

  # Лемматизируем и убираем стоп-слова
  normalized_words = [
    lemmatizer.lemmatize(word) for word in words
    if word and word not in english_stopwords and len(word) > 1
  ]

  return ' '.join(normalized_words)


new_review = st.text_area('Введите отзыв')

test_review = """
я не уверен, что могу порекомендовать этот телефон.
камера не очень, экран тоже тусклый,
но если вы не гонитесь за брендами и лучшими характеристиками -
он несомненно вам подойдет
"""

cleaned_review = normalize_text(new_review)

review_tfidf = vectorizer.transform([cleaned_review])

pred = model.predict(review_tfidf)[0]
score = model.predict_proba(review_tfidf).max()

if new_review:
  st.write("\n" + "-" * 50)
  st.write("\nПредсказание (Logistic Regression):")
  st.write(f"Класс: {pred} | Уверенность: {score:.2f}")


#
# st.header('Рекомендательная модель')
#
# # Загружаем модель
# model_knn_loaded = joblib.load('knn_recommender.pkl')
#
# # Загружаем матрицу
# ratings_sparse_csr_loaded = joblib.load('ratings_sparse_csr.pkl')
#
# # Загружаем item_ids и маппинг
# item_ids_loaded = joblib.load('item_ids.pkl')
# item_to_index_loaded = joblib.load('item_to_index.pkl')
#
# #Загружаем датафрейм
# df_normalized = pd.read_csv('my data/amazon_with_sentiment_reg.csv')
#
#
# def get_recommendations(product_id, n_recommendations=5, target_category=None):
#   if product_id not in item_ids_loaded:
#     st.write(f"Product ID {product_id} не найден.")
#     return []
#
#   idx = item_to_index_loaded[product_id]
#
#   # Получаем вектор товара
#   item_vector = ratings_sparse_csr_loaded[:, idx].T  # (1, n_users)
#
#   # Ищем ближайших соседей (берём больше, чтобы хватило после фильтрации)
#   distances, indices = model_knn_loaded.kneighbors(item_vector, n_neighbors=n_recommendations + 20)
#
#   # Получаем категорию исходного товара (для вывода)
#   original_product_row = df_normalized[df_normalized['product_id'] == product_id].iloc[0]
#   original_category = original_product_row['product_category']
#
#   st.write(f"\nРекомендации для товара '{product_id}' (исходная категория: {original_category})")
#   if target_category:
#     st.write(f"Ищем товары в категории: {target_category}")
#   else:
#     st.write(f"Категория не ограничена")
#
#   recommendations = []
#   for i in range(1, len(distances.flatten())):  # Пропускаем сам товар
#     rec_idx = indices.flatten()[i]
#     rec_product_id = item_ids_loaded[rec_idx]
#
#     # Получаем информацию о кандидате
#     rec_product_row = df_normalized[df_normalized['product_id'] == rec_product_id].iloc[0]
#     rec_category = rec_product_row['product_category']
#
#     # Фильтр по категории (если указана)
#     if target_category and rec_category != target_category:
#       continue
#
#     # Добавляем в рекомендации
#     recommendations.append(rec_product_id)
#     st.write(
#       f"{len(recommendations)}. {rec_product_id} | Категория: {rec_category} | Расстояние: {distances.flatten()[i]:.4f}")
#
#     # Останавливаемся, когда наберём нужное количество
#     if len(recommendations) >= n_recommendations:
#       break
#
#   if not recommendations:
#     st.write("Не найдено подходящих товаров с учётом категории и рейтинга.")
#
#   return recommendations
#
# new_recomendation = st.text_input('Введите ID товара')
#
# test_recommendation = 'B000GIWS7E'
#
# get_recommendations(new_recomendation, n_recommendations=5)