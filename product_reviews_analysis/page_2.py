import streamlit as st
import pandas as pd

st.set_page_config(layout="centered")  # Это возвращает боковые отступы

st.header('А это я вывел табличку из пандаса')

df = pd.DataFrame({
  'first column': [6, 11, 21, 34],
  'second column': [10, 20, 30, 40]
})
st.write(df)

st.line_chart(df)