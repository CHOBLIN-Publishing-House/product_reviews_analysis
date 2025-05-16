logistic_regression_sentiment.pkl — обученная модель
tfidf_vectorizer.pkl — векторизатор TF-IDF

===============
Как загрузить
===============

import joblib

# Загружаем модель
model = joblib.load('logistic_regression_sentiment.pkl')

# Загружаем векторизатор
vectorizer = joblib.load('tfidf_vectorizer.pkl')

=================
Как использовать
=================

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
	
	
new_review = """
я не уверен, что могу порекомендовать этот телефон.
камера не очень, экран тоже тусклый,
но если вы не гонитесь за брендами и лучшими характеристиками -
он несомненно вам подойдет
"""

cleaned_review = normalize_text(new_review)


review_tfidf = vectorizer.transform([cleaned_review])


pred = model.predict(review_tfidf)[0]
score = model.predict_proba(review_tfidf).max()


print("\n" + "-" * 50)
print("ТЕСТОВЫЙ ОТЗЫВ:")
print(new_review.strip())
print("\nПредсказание (Logistic Regression):")
print(f"Класс: {pred} | Уверенность: {score:.2f}")
