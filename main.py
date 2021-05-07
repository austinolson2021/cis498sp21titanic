import streamlit as st
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import pickle
from joblib import dump
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
#from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import time
import os
import fnmatch

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

def train_model():
    datafile= st.file_uploader("Upload csv file", type=["csv"])
    if datafile is not None:
        st.write("File uploaded successfully. Reading the file...")         
        df = pd.read_csv(datafile)
        st.write(df)
        run_model(df)  

def run_model(df):
        if st.button("Create a model"):
            st.write("Running Logistic Regression...");    
            
            X = df.drop('OnboardResp', axis=1);
            y = df['OnboardResp'];
            X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=0);
            std_scaler = StandardScaler();
            X_train_stdscaler = std_scaler.fit_transform(X_train);
            X_test_stdscaler= std_scaler.transform(X_test);

            logreg = LogisticRegression(random_state=0,solver='liblinear').fit(X_train_stdscaler, y_train);
            st.write("Training set accuracy: {:.4f}".format(logreg.score(X_train_stdscaler, y_train)));
            score = logreg.score(X_test_stdscaler, y_test)
            st.write("Test set accuracy: {:.4f}".format(score))
            
            
            st.write("Feature Importances: ");
            #st.write(logreg.coef_);
            
            st.write(pd.DataFrame(zip(X_train.columns, np.transpose(logreg.coef_.tolist()[0])), columns=['features', 'coef']));
            
            y_pred = logreg.predict(X_test_stdscaler);
            
            conf_matrix = confusion_matrix(y_test, y_pred);
            
            TP = conf_matrix[1][1]
            FP = conf_matrix[1][0]
            TN = conf_matrix[0][0]
            FN = conf_matrix[0][1]
            plt.bar(['True Positive','False Positive','True Negative','False Negative'],[TP,FP,TN,FN])
            st.pyplot(plt)
            
            st.write("Confusion Matrix - Actual vs Predicted",conf_matrix);
            
            st.write(classification_report(y_test, y_pred))
            
            #st.write("Classification Report",classification_report(y_test, y_pred));
            
            st.write("Creating AUC-ROC Curve");
            logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test_stdscaler));
            fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test_stdscaler)[:,1]);
            
            plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc);
            
            plt.figure();
            plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc);
            plt.plot([0, 1], [0, 1],'r--');
            plt.xlim([0.0, 1.0]);
            plt.ylim([0.0, 1.05]);
            plt.xlabel('False Positive Rate');
            plt.ylabel('True Positive Rate');
            plt.title('Receiver operating characteristic');
            plt.legend(loc="lower right");
            plt.savefig('Log_ROC');
            st.pyplot(plt);
            
            st.write("AUC: %.3f" % metrics.auc(fpr, tpr));
            st.write("Logistic Regression complete.");
            
            st.write("Saving the model...");
            
            pkl_filename = "pickle_model_" + time.strftime("%Y%m%d-%H%M%S") + ".pkl";
            
            dump(std_scaler, 'scaler.joblib')
            
            tuple_objects =(logreg,score)
            
            #pkl_filename = "pickle_model.pkl";
            pickle.dump(tuple_objects, open(pkl_filename, 'wb'))
                
            st.write("Logistic regression model saved");    

def classify_risk():

    model_list =[]
    for file_name in sorted(os.listdir('./'),reverse=True):
        if fnmatch.fnmatch(file_name,'*.pkl'):
            model_list.append(file_name)

    pickled_model_file= st.selectbox("Load model from the list",model_list)    
    #pickled_model,pickled_score =pickle.load(open("pickle_model_20210503-160534.pkl", 'rb'))
    pickled_model,pickled_score =pickle.load(open(pickled_model_file, 'rb'))
    input_data =get_input_data()

    if st.button("Classify"):
        st.write("Using coefficients from previously saved model...")
        st.write(pickled_model.coef_)
        #inputdata = [[col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15]]; 
        scaler = load("scaler.joblib")
        xtest_scaler = scaler.transform(input_data);        

        y_predict =pickled_model.predict(xtest_scaler);
        y_prob = pickled_model.predict_proba(xtest_scaler)
        if y_predict[0] == 1:
            st.subheader('This risk is SAFE and can be onboarded.')
            st.subheader('Confidence level is {}%'.format(round(y_prob[0,1]*100 , 2)))
        else:
            st.subheader('This risk is CONCERNING. Suggest NOT TO onboard.')
            st.subheader('Confidence level is {}%'.format(round(y_prob[0,0]*100 , 2)))        


def get_input_data():
    col1,col2,col3 = st.beta_columns(3)
    with col1:
        col1 = st.number_input("PropertyOrGLLossesInLast3Years",value=1,min_value=0,max_value=100) 
    with col2: 
        col2 = st.number_input("LeadOrMoldIssues",min_value=0,max_value=100)
    with col3: 
        col3 = st.selectbox("IsTransactionBackDated",(0,1)) 
        
    col4,col5 = st.beta_columns(2)
    with col4:
        col4 = st.selectbox("OccupancyTypeCat",(1,2,3,4))
    with col5: 
        col5 = st.selectbox("ConstructionTypeCat",(1,2,3,4,5)) 
        
    col6,col7= st.beta_columns(2);    
    with col6: 
        col6 = st.number_input("TotalSquareFt",min_value=0,max_value=10000000,value=5000)      
    with col7: 
        col7 = st.number_input("CommercialSQFt",value=1000,min_value=0,max_value=10000000)
        
    col8,col9= st.beta_columns(2)    
    with col8: 
        col8 = st.number_input("YearBuilt",value=2000,min_value=1600,max_value=2021)
    with col9: 
        col9 = st.number_input("NbrOfStories",value=3,min_value=0,max_value=50)       
        
        
    col10,col11= st.beta_columns(2)    
    with col10: 
        col10 = st.number_input("PercentageOccupied",min_value=0,max_value=100,value=100)
    with col11: 
        col11 = st.selectbox("AnyBuildingViolations",(0,1))
        
    col12,col13= st.beta_columns(2)   
    with col12: 
        col12 = st.selectbox("Mercantile",(0,1))
    with col13: 
        col13 = st.selectbox("RenovationGutRehabEverDone",(0,1))        
        
    col14,col15= st.beta_columns(2)   
    with col14: 
        col14 = st.number_input("BuildingLimit",min_value=0,max_value=10000000,value=50000)
    with col15: 
        col15 = st.number_input("RentLimit",min_value=0,max_value=10000000,value=1000)

    
    input_data = [[col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15]]
    return np.array(input_data)


def get_reports():
    report_selection=st.selectbox("Select",("View machine learning models","Other"))
    if report_selection == "View machine learning models":
        st.subheader('Models created from most recent to old')
        index= 1
        for file_name in sorted(os.listdir('./'),reverse=True):
            if fnmatch.fnmatch(file_name,'*.pkl'):
                st.write("Model :" + str(index) )
                st.write(file_name)
                model_name,model_score = pickle.load(open(file_name, 'rb'))
                st.write("M/L algorithm used : " + str(model_name))
                st.write("Coefficents  are : ")
                st.write(model_name.coef_)
                st.write("Score : ") 
                st.write(model_score)
                index +=1


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
            sign_in_check_box = account_tab.checkbox("Log In")
            if sign_in_check_box:
                hashed_sign_in_password = generate_hash(sign_in_password_input)
                result = login_user(sign_in_username_input, check_hashes(sign_in_password_input, hashed_sign_in_password), "users")
                if result:
                    account_tab.success("Logged In As: {}".format(sign_in_username_input))
                    current_user.header(f"{sign_in_username_input}")
                    module_selectbox = st.selectbox("Module Selection", ["Dashboard", "Train Model", "Classify Risk","Reports"])
                    
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
                            #Ravi added 05/03/2021
                            classify_risk()
                            
                    elif module_selectbox == "Train Model":
                        show_risk_model_container = st.beta_container()
                        with show_risk_model_container:
                            st.header("Train Risk Model")
                            #Ravi added 05/03/2021
                            train_model()       
                            
                    elif module_selectbox == "Reports":
                        reports_container = st.beta_container()
                        with reports_container:
                            st.header("Reports")
                            #Ravi added 05/03/2021
                            get_reports()

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