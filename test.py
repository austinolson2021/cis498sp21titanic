# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 17:06:36 2021

@author: austi
"""

import streamlit as st
import pandas as pd
import hashlib
import sqlite3

#create cache oject for the "chat"
@st.cache(allow_output_mutation=True)
def Chat():
    return []

chat=Chat()
name = st.sidebar.text_input("Name")
message = st.sidebar.text_area("Message")
if st.sidebar.button("Post chat message"):
    chat.append((name,message))

try:
    names, messages = zip(*chat)
    chat1 = dict(Name = names, Message =  messages)
    st.table(chat1)
except ValueError:
    st.title("Enter your name and message into the sidebar, and post!")