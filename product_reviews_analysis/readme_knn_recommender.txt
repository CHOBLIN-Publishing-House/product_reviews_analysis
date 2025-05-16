knn_recommender.pkl — обученная модель KNN
ratings_sparse_csr.pkl — разрежённая матрица пользователь x товар
item_ids.pkl — список всех product_id
item_to_index.pkl — словарь для поиска индекса по product_id

===============
Как загрузить
===============

import joblib

# Загружаем модель
model_knn_loaded = joblib.load('knn_recommender.pkl')

# Загружаем матрицу
ratings_sparse_csr_loaded = joblib.load('ratings_sparse_csr.pkl')

# Загружаем item_ids и маппинг
item_ids_loaded = joblib.load('item_ids.pkl')
item_to_index_loaded = joblib.load('item_to_index.pkl')


=================
Как использовать
=================


def get_recommendations(product_id, n_recommendations=5, target_category=None):
    if product_id not in item_ids:
        print(f"Product ID {product_id} не найден.")
        return []

    idx = item_to_index[product_id]

    # Получаем вектор товара
    item_vector = ratings_sparse_csr[:, idx].T  # (1, n_users)

    # Ищем ближайших соседей (берём больше, чтобы хватило после фильтрации)
    distances, indices = model_knn.kneighbors(item_vector, n_neighbors=n_recommendations + 20)

    # Получаем категорию исходного товара (для вывода)
    original_product_row = df_normalized[df_normalized['product_id'] == product_id].iloc[0]
    original_category = original_product_row['product_category']

    print(f"\nРекомендации для товара '{product_id}' (исходная категория: {original_category})")
    if target_category:
        print(f"Ищем товары в категории: {target_category}")
    else:
        print(f"Категория не ограничена")

    recommendations = []
    for i in range(1, len(distances.flatten())):  # Пропускаем сам товар
        rec_idx = indices.flatten()[i]
        rec_product_id = item_ids[rec_idx]

        # Получаем информацию о кандидате
        rec_product_row = df_normalized[df_normalized['product_id'] == rec_product_id].iloc[0]
        rec_category = rec_product_row['product_category']

        # Фильтр по категории (если указана)
        if target_category and rec_category != target_category:
            continue

        # Добавляем в рекомендации
        recommendations.append(rec_product_id)
        print(f"{len(recommendations)}. {rec_product_id} | Категория: {rec_category} | Расстояние: {distances.flatten()[i]:.4f}")

        # Останавливаемся, когда наберём нужное количество
        if len(recommendations) >= n_recommendations:
            break

    if not recommendations:
        print("Не найдено подходящих товаров с учётом категории и рейтинга.")

    return recommendations
    return recommendations
	
get_recommendations('B000GIWS7E', n_recommendations=5)