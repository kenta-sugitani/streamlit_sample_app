#!/usr/bin/env python
# coding: utf-8

import streamlit as st

# if st.button('Say hello'):
#     st.write('Hello World!')

# if st.button('クリックしてください'):
#     st.write('ボタンがクリックされました！')

# x = 10
# 'x: ', x

def convert_number(num):
    binary = bin(num).replace("0b", "")
    hexadecimal = hex(num).replace("0x", "")
    return binary, hexadecimal

st.title('10進数から2進数と16進数への変換')

dec_num = st.number_input('10進数を入力してください', min_value=0, value=0, step=1)

if st.button('変換'):
    binary, hexadecimal = convert_number(dec_num)
    st.write(f'2進数: {binary}')
    st.write(f'16進数: {hexadecimal}')