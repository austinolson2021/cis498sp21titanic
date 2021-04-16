# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 18:25:46 2021

@author: austi
"""

    with account_tab:
        sign_up_button = st.button("Sign Up")
        sign_in_button = st.button("Sign In")
        test = st.selectbox("test", ["one", "two"])

    if sign_in_button:
        sign_in_container = st.beta_container()
        with sign_in_container:
            st.header("Sign In")
            sign_in_username_input = st.text_input("Username", max_chars=16)
            sign_in_password_input = st.text_input("Password", type="password", max_chars=16)
            sign_in_enter_button = st.button("Enter")
            if sign_in_enter_button:
                hashed_sign_in_password = generate_hash(sign_in_password_input)
                st.write(hashed_sign_in_password)
            
    if sign_up_button:
        sign_up_container = st.beta_container()
        with sign_up_container:
            st.header("Sign Up")
            sign_up_username_input = st.text_input("Username", max_chars=16)
            sign_up_password_input = st.text_input("Password", type="password", max_chars=16)
            sign_up_enter_button = st.button("Enter")
            if sign_in_enter_button:
                create_table_if_not_exists('users')