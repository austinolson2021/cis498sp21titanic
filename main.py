import streamlit as st
import pandas as pd
import hashlib
import sqlite3

#Hash Functions
def generate_hash(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if generate_hash(password) == hashed_text:
		return hashed_text
	return False

#DB Connection
conn = sqlite3.connect('data.db')
c = conn.cursor()

#DB Functions
def create_table_if_not_exists(table_name):
	c.execute(f'CREATE TABLE IF NOT EXISTS {table_name}(username TEXT,password TEXT)')


def add_userdata(username,password, table_name):
	c.execute(f'INSERT INTO {table_name}(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data


def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data



def main():
    
    title = st.title("Titanic")
    account_tab = st.sidebar.beta_expander("Accounts")
    
    account_selectbox = account_tab.selectbox("", ["Sign In", "Sign Up"])

    if account_selectbox == "Sign In":
        sign_in_container = st.beta_container()
        with sign_in_container:
            st.header("Sign In")
            sign_in_username_input = st.text_input("Username", max_chars=16)
            sign_in_password_input = st.text_input("Password", type="password", max_chars=16)
            sign_in_enter_button = st.button("Enter")
            if sign_in_enter_button:
                hashed_sign_in_password = generate_hash(sign_in_password_input)
                result = login_user(sign_in_username_input, check_hashes(sign_in_password_input, hashed_sign_in_password))
                if result:
                    st.success("Logged In as {}".format(sign_in_username_input))
                else:
                    st.warning("Incorrect Username/Password")
            
    if account_selectbox == "Sign Up":
        sign_up_container = st.beta_container()
        with sign_up_container:
            st.header("Sign Up")
            sign_up_username_input = st.text_input("Username", max_chars=16)
            sign_up_password_input = st.text_input("Password", type="password", max_chars=16)
            sign_up_enter_button = st.button("Enter")
            if sign_up_enter_button:
                create_table_if_not_exists('users')
                add_userdata(sign_up_username_input, generate_hash(sign_up_password_input), 'users')
                st.success("You have successfully created a valid account.")
                st.info("Go To Sign In Menu")

if __name__ == '__main__':
	main()