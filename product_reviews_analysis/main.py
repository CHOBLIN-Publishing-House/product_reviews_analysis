import streamlit as st

page_1 = st.Page("page_1.py", title="Аналитика")
page_2 = st.Page("page_2.py", title="Прогноз")


# Set up navigation
pg = st.navigation([page_1, page_2])

# Run the selected page
pg.run()
