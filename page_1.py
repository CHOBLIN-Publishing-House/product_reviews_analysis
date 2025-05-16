import streamlit as st

st.set_page_config(layout="wide")  # Это убирает боковые отступы


st.title("Дашборд Yandex Datalens")

st.markdown("""
<iframe 
    src="https://datalens.yandex/fqptgydqi0by1?_no_controls=1" 
    width="100%" 
    height="600px" 
    frameborder="0">
</iframe>
""", unsafe_allow_html=True)

