import numpy as np
import pandas as pd
import warnings
import pickle
 
warnings.filterwarnings("ignore")
    
def process_data():
        
    data = pd.read_csv(".\\temp\\input.csv", index_col = 0)
    #data = pd.read_csv(r"C:\Users\ayan.tiwari\Desktop\Hackathon\data_inputs_test.csv", index_col = 0)
    #data_targets = pd.read_csv(".\\temp\\data_targets_test.csv", index_col = 0,  header = None)
    
    credit_data = data.copy()
    #credit_data_targets = data_targets.copy()
    
    output_data = credit_data
    
    credit_data['emp_length'].unique()
    
    credit_data['emp_length_int'] = credit_data['emp_length'].str.replace('\+ years', '')
    credit_data['emp_length_int'] = credit_data['emp_length_int'].str.replace('< 1 year', str(0))
    credit_data['emp_length_int'] = credit_data['emp_length_int'].str.replace('n/a',  str(0))
    credit_data['emp_length_int'] = credit_data['emp_length_int'].str.replace(' years', '')
    credit_data['emp_length_int'] = credit_data['emp_length_int'].str.replace(' year', '')
    
    credit_data['emp_length_int'] = pd.to_numeric(credit_data['emp_length_int'])
    
    credit_data['earliest_cr_line_date'] = pd.to_datetime(credit_data['earliest_cr_line'], format = '%b-%y')
    
    pd.to_datetime('2019-12-01') - credit_data['earliest_cr_line_date']
    
    credit_data['mths_since_earliest_cr_line'] = round(pd.to_numeric((pd.to_datetime('2019-12-01') - credit_data['earliest_cr_line_date']) / np.timedelta64(1, 'M')))
    
    credit_data['mths_since_earliest_cr_line'] = round(pd.to_numeric((pd.to_datetime('2019-12-01') - credit_data['earliest_cr_line_date']) / np.timedelta64(1, 'M')))
    
    credit_data['mths_since_earliest_cr_line'].describe()
    
    credit_data.loc[: , ['earliest_cr_line', 'earliest_cr_line_date', 'mths_since_earliest_cr_line']][credit_data['mths_since_earliest_cr_line'] < 0]
    
    credit_data.loc[ : , ['mths_since_earliest_cr_line']][credit_data['mths_since_earliest_cr_line'] < 0] = credit_data['mths_since_earliest_cr_line'].max()
    
    credit_data['term']
    
    credit_data['term_int'] = credit_data['term'].str.replace(' months', '')
    
    credit_data['term_int'] = pd.to_numeric(credit_data['term'].str.replace(' months', ''))
    
    credit_data['issue_d_date'] = pd.to_datetime(credit_data['issue_d'], format = '%b-%y')
    
    credit_data['mths_since_issue_d'] = round(pd.to_numeric((pd.to_datetime('2019-12-01') - credit_data['issue_d_date']) / np.timedelta64(1, 'M')))
    
    # ### Preprocessing few discrete variables
    
    credit_data_dummies = [pd.get_dummies(credit_data['grade'], prefix = 'grade', prefix_sep = ':'),
                         pd.get_dummies(credit_data['home_ownership'], prefix = 'home_ownership', prefix_sep = ':'),
                         pd.get_dummies(credit_data['verification_status'], prefix = 'verification_status', prefix_sep = ':'),
                         pd.get_dummies(credit_data['loan_status'], prefix = 'loan_status', prefix_sep = ':')]
    
    credit_data_dummies = pd.concat(credit_data_dummies, axis = 1)
    
    credit_data = pd.concat([credit_data, credit_data_dummies], axis = 1)
    
    credit_data.isnull().sum()
    
    credit_data['total_rev_hi_lim'].fillna(credit_data['funded_amnt'], inplace=True)
    
    credit_data['total_rev_hi_lim'].isnull().sum()
    
    credit_data['annual_inc'].fillna(credit_data['annual_inc'].mean(), inplace=True)
    
    credit_data['mths_since_earliest_cr_line'].fillna(0, inplace=True)
    credit_data['acc_now_delinq'].fillna(0, inplace=True)
    credit_data['total_acc'].fillna(0, inplace=True)
    credit_data['pub_rec'].fillna(0, inplace=True)
    credit_data['open_acc'].fillna(0, inplace=True)
    credit_data['inq_last_6mths'].fillna(0, inplace=True)
    credit_data['delinq_2yrs'].fillna(0, inplace=True)
    credit_data['emp_length_int'].fillna(0, inplace=True)
    # We fill the missing values with zeroes.
    
    # ### PD Model
    
    df_inputs_prepr = credit_data
        
    import matplotlib.pyplot as plt
        
    df_inputs_prepr['term:36'] = np.where((df_inputs_prepr['term_int'] == 36), 1, 0)
    df_inputs_prepr['term:60'] = np.where((df_inputs_prepr['term_int'] == 60), 1, 0)
    
    
    df_inputs_prepr['emp_length:0'] = np.where(df_inputs_prepr['emp_length_int'].isin([0]), 1, 0)
    df_inputs_prepr['emp_length:1'] = np.where(df_inputs_prepr['emp_length_int'].isin([1]), 1, 0)
    df_inputs_prepr['emp_length:2-4'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(2, 5)), 1, 0)
    df_inputs_prepr['emp_length:5-6'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(5, 7)), 1, 0)
    df_inputs_prepr['emp_length:7-9'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(7, 10)), 1, 0)
    df_inputs_prepr['emp_length:10'] = np.where(df_inputs_prepr['emp_length_int'].isin([10]), 1, 0)
    
    df_inputs_prepr['int_rate_factor'] = pd.cut(df_inputs_prepr['int_rate'], 50)
    
    
    df_inputs_prepr['int_rate:<9.548'] = np.where((df_inputs_prepr['int_rate'] <= 9.548), 1, 0)
    df_inputs_prepr['int_rate:9.548-12.025'] = np.where((df_inputs_prepr['int_rate'] > 9.548) & (df_inputs_prepr['int_rate'] <= 12.025), 1, 0)
    df_inputs_prepr['int_rate:12.025-15.74'] = np.where((df_inputs_prepr['int_rate'] > 12.025) & (df_inputs_prepr['int_rate'] <= 15.74), 1, 0)
    df_inputs_prepr['int_rate:15.74-20.281'] = np.where((df_inputs_prepr['int_rate'] > 15.74) & (df_inputs_prepr['int_rate'] <= 20.281), 1, 0)
    df_inputs_prepr['int_rate:>20.281'] = np.where((df_inputs_prepr['int_rate'] > 20.281), 1, 0)
    
    df_inputs_prepr['mths_since_earliest_cr_line_factor'] = pd.cut(df_inputs_prepr['mths_since_earliest_cr_line'], 50)
    
    
    df_inputs_prepr['mths_since_earliest_cr_line:<140'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(140)), 1, 0)
    df_inputs_prepr['mths_since_earliest_cr_line:141-164'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(140, 165)), 1, 0)
    df_inputs_prepr['mths_since_earliest_cr_line:165-247'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(165, 248)), 1, 0)
    df_inputs_prepr['mths_since_earliest_cr_line:248-270'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(248, 271)), 1, 0)
    df_inputs_prepr['mths_since_earliest_cr_line:271-352'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(271, 353)), 1, 0)
    df_inputs_prepr['mths_since_earliest_cr_line:>352'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(353, int(df_inputs_prepr['mths_since_earliest_cr_line'].max()))), 1, 0)
    
    
    df_inputs_prepr['delinq_2yrs:0'] = np.where((df_inputs_prepr['delinq_2yrs'] == 0), 1, 0)
    df_inputs_prepr['delinq_2yrs:1-3'] = np.where((df_inputs_prepr['delinq_2yrs'] >= 1) & (df_inputs_prepr['delinq_2yrs'] <= 3), 1, 0)
    df_inputs_prepr['delinq_2yrs:>=4'] = np.where((df_inputs_prepr['delinq_2yrs'] >= 9), 1, 0)
    
    
    df_inputs_prepr['inq_last_6mths:0'] = np.where((df_inputs_prepr['inq_last_6mths'] == 0), 1, 0)
    df_inputs_prepr['inq_last_6mths:1-2'] = np.where((df_inputs_prepr['inq_last_6mths'] >= 1) & (df_inputs_prepr['inq_last_6mths'] <= 2), 1, 0)
    df_inputs_prepr['inq_last_6mths:3-6'] = np.where((df_inputs_prepr['inq_last_6mths'] >= 3) & (df_inputs_prepr['inq_last_6mths'] <= 6), 1, 0)
    df_inputs_prepr['inq_last_6mths:>6'] = np.where((df_inputs_prepr['inq_last_6mths'] > 6), 1, 0)
    
    
    df_inputs_prepr['open_acc:0'] = np.where((df_inputs_prepr['open_acc'] == 0), 1, 0)
    df_inputs_prepr['open_acc:1-3'] = np.where((df_inputs_prepr['open_acc'] >= 1) & (df_inputs_prepr['open_acc'] <= 3), 1, 0)
    df_inputs_prepr['open_acc:4-12'] = np.where((df_inputs_prepr['open_acc'] >= 4) & (df_inputs_prepr['open_acc'] <= 12), 1, 0)
    df_inputs_prepr['open_acc:13-17'] = np.where((df_inputs_prepr['open_acc'] >= 13) & (df_inputs_prepr['open_acc'] <= 17), 1, 0)
    df_inputs_prepr['open_acc:18-22'] = np.where((df_inputs_prepr['open_acc'] >= 18) & (df_inputs_prepr['open_acc'] <= 22), 1, 0)
    df_inputs_prepr['open_acc:23-25'] = np.where((df_inputs_prepr['open_acc'] >= 23) & (df_inputs_prepr['open_acc'] <= 25), 1, 0)
    df_inputs_prepr['open_acc:26-30'] = np.where((df_inputs_prepr['open_acc'] >= 26) & (df_inputs_prepr['open_acc'] <= 30), 1, 0)
    df_inputs_prepr['open_acc:>=31'] = np.where((df_inputs_prepr['open_acc'] >= 31), 1, 0)
    
    
    df_inputs_prepr['pub_rec:0-2'] = np.where((df_inputs_prepr['pub_rec'] >= 0) & (df_inputs_prepr['pub_rec'] <= 2), 1, 0)
    df_inputs_prepr['pub_rec:3-4'] = np.where((df_inputs_prepr['pub_rec'] >= 3) & (df_inputs_prepr['pub_rec'] <= 4), 1, 0)
    df_inputs_prepr['pub_rec:>=5'] = np.where((df_inputs_prepr['pub_rec'] >= 5), 1, 0)
    
    df_inputs_prepr['total_acc_factor'] = pd.cut(df_inputs_prepr['total_acc'], 50)
    
    
    df_inputs_prepr['total_acc:<=27'] = np.where((df_inputs_prepr['total_acc'] <= 27), 1, 0)
    df_inputs_prepr['total_acc:28-51'] = np.where((df_inputs_prepr['total_acc'] >= 28) & (df_inputs_prepr['total_acc'] <= 51), 1, 0)
    df_inputs_prepr['total_acc:>=52'] = np.where((df_inputs_prepr['total_acc'] >= 52), 1, 0)
    
    
    df_inputs_prepr['acc_now_delinq:0'] = np.where((df_inputs_prepr['acc_now_delinq'] == 0), 1, 0)
    df_inputs_prepr['acc_now_delinq:>=1'] = np.where((df_inputs_prepr['acc_now_delinq'] >= 1), 1, 0)
    
    df_inputs_prepr['total_rev_hi_lim_factor'] = pd.cut(df_inputs_prepr['total_rev_hi_lim'], 2000)
    
    df_inputs_prepr['total_rev_hi_lim:<=5K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] <= 5000), 1, 0)
    df_inputs_prepr['total_rev_hi_lim:5K-10K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 5000) & (df_inputs_prepr['total_rev_hi_lim'] <= 10000), 1, 0)
    df_inputs_prepr['total_rev_hi_lim:10K-20K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 10000) & (df_inputs_prepr['total_rev_hi_lim'] <= 20000), 1, 0)
    df_inputs_prepr['total_rev_hi_lim:20K-30K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 20000) & (df_inputs_prepr['total_rev_hi_lim'] <= 30000), 1, 0)
    df_inputs_prepr['total_rev_hi_lim:30K-40K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 30000) & (df_inputs_prepr['total_rev_hi_lim'] <= 40000), 1, 0)
    df_inputs_prepr['total_rev_hi_lim:40K-55K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 40000) & (df_inputs_prepr['total_rev_hi_lim'] <= 55000), 1, 0)
    df_inputs_prepr['total_rev_hi_lim:55K-95K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 55000) & (df_inputs_prepr['total_rev_hi_lim'] <= 95000), 1, 0)
    df_inputs_prepr['total_rev_hi_lim:>95K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 95000), 1, 0)
    
    df_inputs_prepr['installment_factor'] = pd.cut(df_inputs_prepr['installment'], 50)
    
    df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['annual_inc'] <= 140000 , :]
    df_inputs_prepr_temp['annual_inc_factor'] = pd.cut(df_inputs_prepr_temp['annual_inc'], 50)
    
    df_inputs_prepr['annual_inc:<20K'] = np.where((df_inputs_prepr['annual_inc'] <= 20000), 1, 0)
    df_inputs_prepr['annual_inc:20K-30K'] = np.where((df_inputs_prepr['annual_inc'] > 20000) & (df_inputs_prepr['annual_inc'] <= 30000), 1, 0)
    df_inputs_prepr['annual_inc:30K-40K'] = np.where((df_inputs_prepr['annual_inc'] > 30000) & (df_inputs_prepr['annual_inc'] <= 40000), 1, 0)
    df_inputs_prepr['annual_inc:40K-50K'] = np.where((df_inputs_prepr['annual_inc'] > 40000) & (df_inputs_prepr['annual_inc'] <= 50000), 1, 0)
    df_inputs_prepr['annual_inc:50K-60K'] = np.where((df_inputs_prepr['annual_inc'] > 50000) & (df_inputs_prepr['annual_inc'] <= 60000), 1, 0)
    df_inputs_prepr['annual_inc:60K-70K'] = np.where((df_inputs_prepr['annual_inc'] > 60000) & (df_inputs_prepr['annual_inc'] <= 70000), 1, 0)
    df_inputs_prepr['annual_inc:70K-80K'] = np.where((df_inputs_prepr['annual_inc'] > 70000) & (df_inputs_prepr['annual_inc'] <= 80000), 1, 0)
    df_inputs_prepr['annual_inc:80K-90K'] = np.where((df_inputs_prepr['annual_inc'] > 80000) & (df_inputs_prepr['annual_inc'] <= 90000), 1, 0)
    df_inputs_prepr['annual_inc:90K-100K'] = np.where((df_inputs_prepr['annual_inc'] > 90000) & (df_inputs_prepr['annual_inc'] <= 100000), 1, 0)
    df_inputs_prepr['annual_inc:100K-120K'] = np.where((df_inputs_prepr['annual_inc'] > 100000) & (df_inputs_prepr['annual_inc'] <= 120000), 1, 0)
    df_inputs_prepr['annual_inc:120K-140K'] = np.where((df_inputs_prepr['annual_inc'] > 120000) & (df_inputs_prepr['annual_inc'] <= 140000), 1, 0)
    df_inputs_prepr['annual_inc:>140K'] = np.where((df_inputs_prepr['annual_inc'] > 140000), 1, 0)
    
    
    df_inputs_prepr['dti:<=1.4'] = np.where((df_inputs_prepr['dti'] <= 1.4), 1, 0)
    df_inputs_prepr['dti:1.4-3.5'] = np.where((df_inputs_prepr['dti'] > 1.4) & (df_inputs_prepr['dti'] <= 3.5), 1, 0)
    df_inputs_prepr['dti:3.5-7.7'] = np.where((df_inputs_prepr['dti'] > 3.5) & (df_inputs_prepr['dti'] <= 7.7), 1, 0)
    df_inputs_prepr['dti:7.7-10.5'] = np.where((df_inputs_prepr['dti'] > 7.7) & (df_inputs_prepr['dti'] <= 10.5), 1, 0)
    df_inputs_prepr['dti:10.5-16.1'] = np.where((df_inputs_prepr['dti'] > 10.5) & (df_inputs_prepr['dti'] <= 16.1), 1, 0)
    df_inputs_prepr['dti:16.1-20.3'] = np.where((df_inputs_prepr['dti'] > 16.1) & (df_inputs_prepr['dti'] <= 20.3), 1, 0)
    df_inputs_prepr['dti:20.3-21.7'] = np.where((df_inputs_prepr['dti'] > 20.3) & (df_inputs_prepr['dti'] <= 21.7), 1, 0)
    df_inputs_prepr['dti:21.7-22.4'] = np.where((df_inputs_prepr['dti'] > 21.7) & (df_inputs_prepr['dti'] <= 22.4), 1, 0)
    df_inputs_prepr['dti:22.4-35'] = np.where((df_inputs_prepr['dti'] > 22.4) & (df_inputs_prepr['dti'] <= 35), 1, 0)
    df_inputs_prepr['dti:>35'] = np.where((df_inputs_prepr['dti'] > 35), 1, 0)
    
    
    ### ===========================================================================================================
    
    
    ### ===========================================================================================================
    
    with open(".\\my_model.pkl", "rb") as f:
    #with open(r"C:\Users\ayan.tiwari\Desktop\Hackathon\my_model.pkl", "rb") as f:    
            m = pickle.load(f)
    
    # ### Model Validation
    
    # Here, from the dataframe with inputs for testing, we keep the same variables that we used in our final PD model.
    inputs_test_with_ref_cat = df_inputs_prepr.loc[: , ['grade:A',
    'grade:B',
    'grade:C',
    'grade:D',
    'grade:E',
    'grade:F',
    'grade:G',
    #'home_ownership:RENT_OTHER_NONE_ANY',
    'home_ownership:OWN',
    'home_ownership:MORTGAGE',
    'verification_status:Not Verified',
    'verification_status:Source Verified',
    'verification_status:Verified',
    'term:36',
    'term:60',
    'emp_length:0',
    'emp_length:1',
    'emp_length:2-4',
    'emp_length:5-6',
    'emp_length:7-9',
    'emp_length:10',
    'int_rate:<9.548',
    'int_rate:9.548-12.025',
    'int_rate:12.025-15.74',
    'int_rate:15.74-20.281',
    'int_rate:>20.281',
    'mths_since_earliest_cr_line:<140',
    'mths_since_earliest_cr_line:141-164',
    'mths_since_earliest_cr_line:165-247',
    'mths_since_earliest_cr_line:248-270',
    'mths_since_earliest_cr_line:271-352',
    'mths_since_earliest_cr_line:>352',
    'delinq_2yrs:0',
    'delinq_2yrs:1-3',
    'delinq_2yrs:>=4',
    'inq_last_6mths:0',
    'inq_last_6mths:1-2',
    'inq_last_6mths:3-6',
    'inq_last_6mths:>6',
    'open_acc:0',
    'open_acc:1-3',
    'open_acc:4-12',
    'open_acc:13-17',
    'open_acc:18-22',
    'open_acc:23-25',
    'open_acc:26-30',
    'open_acc:>=31',
    'pub_rec:0-2',
    'pub_rec:3-4',
    'pub_rec:>=5',
    'total_acc:<=27',
    'total_acc:28-51',
    'total_acc:>=52',
    'acc_now_delinq:0',
    'acc_now_delinq:>=1',
    'total_rev_hi_lim:<=5K',
    'total_rev_hi_lim:5K-10K',
    'total_rev_hi_lim:10K-20K',
    'total_rev_hi_lim:20K-30K',
    'total_rev_hi_lim:30K-40K',
    'total_rev_hi_lim:40K-55K',
    'total_rev_hi_lim:55K-95K',
    'total_rev_hi_lim:>95K',
    'annual_inc:<20K',
    'annual_inc:20K-30K',
    'annual_inc:30K-40K',
    'annual_inc:40K-50K',
    'annual_inc:50K-60K',
    'annual_inc:60K-70K',
    'annual_inc:70K-80K',
    'annual_inc:80K-90K',
    'annual_inc:90K-100K',
    'annual_inc:100K-120K',
    'annual_inc:120K-140K',
    'annual_inc:>140K',
    'dti:<=1.4',
    'dti:1.4-3.5',
    'dti:3.5-7.7',
    'dti:7.7-10.5',
    'dti:10.5-16.1',
    'dti:16.1-20.3',
    'dti:20.3-21.7',
    'dti:21.7-22.4',
    'dti:22.4-35',
    'dti:>35',
    ]]
    
    # And here, in the list below, we keep the variable names for the reference categories,
    # only for the variables we used in our final PD model.
    ref_categories = ['grade:G',
    #'home_ownership:RENT_OTHER_NONE_ANY',
    'verification_status:Verified',
    'term:60',
    'emp_length:0',
    'int_rate:>20.281',
    'mths_since_earliest_cr_line:<140',
    #'delinq_2yrs:>=4',
    'inq_last_6mths:>6',
    #'open_acc:0',
    #'pub_rec:0-2',
    #'total_acc:<=27',
    'acc_now_delinq:0',
    #'total_rev_hi_lim:<=5K',
    'annual_inc:<20K',
    'dti:>35']
    
    inputs_test = inputs_test_with_ref_cat.drop(ref_categories, axis = 1)
    
    inputs_test['grade:A'].fillna(0, inplace = True)
    inputs_test['grade:B'].fillna(0, inplace = True)
    inputs_test['grade:C'].fillna(0, inplace = True)
    inputs_test['grade:D'].fillna(0, inplace = True)
    inputs_test['grade:E'].fillna(0, inplace = True)
    inputs_test['grade:F'].fillna(0, inplace = True)
    inputs_test['home_ownership:OWN'].fillna(0, inplace = True)
    inputs_test['home_ownership:MORTGAGE'].fillna(0, inplace = True)
    inputs_test['verification_status:Not Verified'].fillna(0, inplace = True)
    inputs_test['verification_status:Source Verified'].fillna(0, inplace = True)
    
    y_hat_test = m.predict(inputs_test)
    # Calculates the predicted values for the dependent variable (targets)
    # based on the values of the independent variables (inputs) supplied as an argument.
    
    y_hat_test_proba = m.predict_proba(inputs_test)
    # Calculates the predicted probability values for the dependent variable (targets)
    # based on the values of the independent variables (inputs) supplied as an argument.
    
    y_hat_test_proba
    # This is an array of arrays of predicted class probabilities for all classes.
    # In this case, the first value of every sub-array is the probability for the observation to belong to the first class, i.e. 0,
    # and the second value is the probability for the observation to belong to the first class, i.e. 1.
    
    y_hat_test_proba[:][:,1]
    # Here we take all the arrays in the array, and from each array, we take all rows, and only the element with index 1,
    # that is, the second element.
    # In other words, we take only the probabilities for being 1.
    
    y_hat_test_proba = y_hat_test_proba[: ][: , 1]
    # We store these probabilities in a variable.
    
    y_hat_test_proba
    # This variable contains an array of probabilities of being 1.
    
     # Same as above.
    feature_name = inputs_test.columns.values
    summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
    summary_table['Coefficients'] = np.transpose(m.coef_)
    summary_table.index = summary_table.index + 1
    summary_table.loc[0] = ['Intercept', m.intercept_[0]]
    summary_table = summary_table.sort_index()
    summary_table
    
    # ### Calculating Credit Scores
    
    df_ref_categories = pd.DataFrame(ref_categories, columns = ['Feature name'])
    # We create a new dataframe with one column. Its values are the values from the 'reference_categories' list.
    # We name it 'Feature name'.
    df_ref_categories['Coefficients'] = 0
    # We create a second column, called 'Coefficients', which contains only 0 values.
    df_ref_categories['p_values'] = np.nan
    # We create a third column, called 'p_values', with contains only NaN values.
    df_ref_categories
    
    df_scorecard = pd.concat([summary_table, df_ref_categories])
    # Concatenates two dataframes.
    df_scorecard = df_scorecard.reset_index()
    # We reset the index of a dataframe.
    df_scorecard
    
    df_scorecard['Original feature name'] = df_scorecard['Feature name'].str.split(':').str[0]
    # We create a new column, called 'Original feature name', which contains the value of the 'Feature name' column,
    # up to the column symbol.
    df_scorecard
    
    min_score = 300
    max_score = 850
    
    df_scorecard.groupby('Original feature name')['Coefficients'].min()
    # Groups the data by the values of the 'Original feature name' column.
    # Aggregates the data in the 'Coefficients' column, calculating their minimum.
    
    min_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].min().sum()
    # Up to the 'min()' method everything is the same as in te line above.
    # Then, we aggregate further and sum all the minimum values.
    min_sum_coef
    
    df_scorecard.groupby('Original feature name')['Coefficients'].max()
    # Groups the data by the values of the 'Original feature name' column.
    # Aggregates the data in the 'Coefficients' column, calculating their maximum.
    
    max_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].max().sum()
    # Up to the 'min()' method everything is the same as in te line above.
    # Then, we aggregate further and sum all the maximum values.
    max_sum_coef
    
    df_scorecard['Score - Calculation'] = df_scorecard['Coefficients'] * (max_score - min_score) / (max_sum_coef - min_sum_coef)
    # We multiply the value of the 'Coefficients' column by the ration of the differences between
    # maximum score and minimum score and maximum sum of coefficients and minimum sum of cefficients.
    df_scorecard
    
    df_scorecard['Score - Calculation'][0] = ((df_scorecard['Coefficients'][0] - min_sum_coef) / (max_sum_coef - min_sum_coef)) * (max_score - min_score) + min_score
    # We divide the difference of the value of the 'Coefficients' column and the minimum sum of coefficients by
    # the difference of the maximum sum of coefficients and the minimum sum of coefficients.
    # Then, we multiply that by the difference between the maximum score and the minimum score.
    # Then, we add minimum score. 
    df_scorecard
    
    df_scorecard['Score - Preliminary'] = df_scorecard['Score - Calculation'].round()
    # We round the values of the 'Score - Calculation' column.
    df_scorecard
    
    min_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Preliminary'].min().sum()
    # Groups the data by the values of the 'Original feature name' column.
    # Aggregates the data in the 'Coefficients' column, calculating their minimum.
    # Sums all minimum values.
    min_sum_score_prel
    
    max_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Preliminary'].max().sum()
    # Groups the data by the values of the 'Original feature name' column.
    # Aggregates the data in the 'Coefficients' column, calculating their maximum.
    # Sums all maximum values.
    max_sum_score_prel
    
    # ### Calculating Credit Score
    
    inputs_test_with_ref_cat_w_intercept = inputs_test_with_ref_cat
    
    inputs_test_with_ref_cat_w_intercept['grade:A'].fillna(0, inplace = True)
    inputs_test_with_ref_cat_w_intercept['grade:B'].fillna(0, inplace = True)
    inputs_test_with_ref_cat_w_intercept['grade:C'].fillna(0, inplace = True)
    inputs_test_with_ref_cat_w_intercept['grade:D'].fillna(0, inplace = True)
    inputs_test_with_ref_cat_w_intercept['grade:E'].fillna(0, inplace = True)
    inputs_test_with_ref_cat_w_intercept['grade:F'].fillna(0, inplace = True)
    inputs_test_with_ref_cat_w_intercept['grade:G'].fillna(0, inplace = True)
    inputs_test_with_ref_cat_w_intercept['home_ownership:OWN'].fillna(0, inplace = True)
    inputs_test_with_ref_cat_w_intercept['home_ownership:MORTGAGE'].fillna(0, inplace = True)
    inputs_test_with_ref_cat_w_intercept['verification_status:Not Verified'].fillna(0, inplace = True)
    inputs_test_with_ref_cat_w_intercept['verification_status:Source Verified'].fillna(0, inplace = True)
    
    
    inputs_test_with_ref_cat_w_intercept.insert(0, 'Intercept', 1)
    # We insert a column in the dataframe, with an index of 0, that is, in the beginning of the dataframe.
    # The name of that column is 'Intercept', and its values are 1s.
    
    inputs_test_with_ref_cat_w_intercept = inputs_test_with_ref_cat_w_intercept[df_scorecard['Feature name'].values]
    # Here, from the 'inputs_test_with_ref_cat_w_intercept' dataframe, we keep only the columns with column names,
    # exactly equal to the row values of the 'Feature name' column from the 'df_scorecard' dataframe.
    
    scorecard_scores = df_scorecard['Score - Preliminary']
    
    scorecard_scores = scorecard_scores.values.reshape(85, 1)
    
    y_scores = inputs_test_with_ref_cat_w_intercept.dot(scorecard_scores)
    # Here we multiply the values of each row of the dataframe by the values of each column of the variable,
    # which is an argument of the 'dot' method, and sum them. It's essentially the sum of the products.
    
    # ### Going back to PD from Credit Scores
    
    sum_coef_from_score = ((y_scores - min_score) / (max_score - min_score)) * (max_sum_coef - min_sum_coef) + min_sum_coef
    # We divide the difference between the scores and the minimum score by
    # the difference between the maximum score and the minimum score.
    # Then, we multiply that by the difference between the maximum sum of coefficients and the minimum sum of coefficients.
    # Then, we add the minimum sum of coefficients.
    
    y_hat_proba_from_score = np.exp(sum_coef_from_score) / (np.exp(sum_coef_from_score) + 1)
    # Here we divide an exponent raised to sum of coefficients from score by
    # an exponent raised to sum of coefficients from score plus one.
    
    output_data.insert(0, 'Credit_Score', 1)
    output_data['Credit_Score'] = y_scores
    
    output_data.to_csv(".\\temp\\export.csv")
