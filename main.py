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
conn = sqlite3.connect('database.db')
c = conn.cursor()

#DB Functions
def create_table_if_not_exists(table_name):
    c.execute(f'CREATE TABLE IF NOT EXISTS {table_name}(username TEXT,password TEXT)')


def add_userdata(username, password, table_name):
    c.execute(f'INSERT INTO {table_name}(username,password) VALUES (?,?)',(username,password))
    conn.commit()

def check_if_user_exists(username, table_name):
    c.execute(f'SELECT EXISTS(SELECT * FROM {table_name} WHERE username = ?)', (username,))
    [exists] = c.fetchone()
    if exists:
        return True
    else:
        return False

def login_user(username,password,table_name):
	c.execute(f'SELECT * FROM {table_name} WHERE username = ? AND password = ?',(username,password))
	data = c.fetchall()
	return data


def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data



def main():
    
    title = st.title("Titanic")
    navigation_pane = st.sidebar
    current_user = navigation_pane.header("")
    account_tab = navigation_pane.beta_expander("Account")
    account_selectbox = account_tab.selectbox("", ["Sign In", "Sign Up"])
    if account_selectbox == "Sign In":
            account_tab.header("Sign In")
            sign_in_username_input = account_tab.text_input("Username", max_chars=16)
            sign_in_password_input = account_tab.text_input("Password", type="password", max_chars=16)
            sign_in_check_box = account_tab.checkbox("Enter")
            if sign_in_check_box:
                hashed_sign_in_password = generate_hash(sign_in_password_input)
                result = login_user(sign_in_username_input, check_hashes(sign_in_password_input, hashed_sign_in_password), "users")
                if result:
                    account_tab.success("Logged In as {}".format(sign_in_username_input))
                    current_user.header(f"{sign_in_username_input}")
                    module_selectbox = st.selectbox("Module Selection", ["Dashboard", "Classify Risk", "Show Risk Model", "Reports"])
                    
                    if module_selectbox == "Dashboard":
                        dashboard_container = st.beta_container()
                        with dashboard_container:
                            st.header("Dashboard")
                            onboarded_risks_container = st.beta_container()
                            with onboarded_risks_container:
                                st.write("Onboarded Risks")
                                st.table()
                            col1, col2 = st.beta_columns(2)
                            with col1:
                                st.write("Flagged Risks With Premium")
                                st.table()
                            with col2:
                                st.write("Onboarded Premium/Revenue")
                                st.table()
                            
                    elif module_selectbox == "Classify Risk":
                        classify_risk_container = st.beta_container()
                        with classify_risk_container:
                            st.header("Classify Risk")
                            st.write("Put your work here")
                            
                    elif module_selectbox == "Show Risk Model":
                        show_risk_model_container = st.beta_container()
                        with show_risk_model_container:
                            st.header("Show Risk Model")
                            st.write("Put your work here")
                            
                    elif module_selectbox == "Reports":
                        reports_container = st.beta_container()
                        with reports_container:
                            st.header("Reports")
                            st.write("Put your work here")
                else:
                    st.warning("Incorrect Username/Password")
            
    elif account_selectbox == "Sign Up":
        account_tab.header("Sign Up")
        sign_up_username_input = account_tab.text_input("Username", max_chars=16)
        sign_up_password_input = account_tab.text_input("Password", type="password", max_chars=16)
        sign_up_enter_button = account_tab.button("Enter")
        if sign_up_enter_button:
            create_table_if_not_exists('users')
            user_exists = check_if_user_exists(sign_up_username_input, 'users')
            if user_exists == True:
                account_tab.warning("Username Already Exists")
            else:
                add_userdata(sign_up_username_input, generate_hash(sign_up_password_input), 'users')
                account_tab.success("You have successfully created a valid account")
                account_tab.info("Go To Sign In Menu")
                    
if __name__ == '__main__':
	main()