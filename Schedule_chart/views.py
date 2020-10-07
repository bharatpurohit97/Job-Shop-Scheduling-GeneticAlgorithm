from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from production_plan_project.settings import BASE_DIR
from .data_json import  data_json

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import chart_studio.plotly as py
# import plotly.figure_factory as ff
import datetime
import time
import copy

import openpyxl as xl
import json
import logging as ls


def static_data(json_file):
    with open(json_file) as f:
        data = json.load(f)

    return data


def from_excel_to_json(sheet_name):
    '''
    please provide excel sheet
    '''
    data = {}
    wb = xl.load_workbook(sheet_name,data_only=True)
    sheet_names = wb.sheetnames
    
    for name in sheet_names:
        wb_sheet = wb[name]
        
        
        cell_values = wb_sheet.values
        df = pd.DataFrame(cell_values,columns=next(cell_values))
        
        #removing uwanted chars
        df.iloc[:,0] = df.iloc[:,0].apply(lambda x : x.strip())
        #set index
        df.index = df.iloc[:,0]
        #removing extra column
        df.drop(columns=df.columns[0],inplace= True)
        
        data[name] = df.T.to_dict()
        
    return data


def from_json_to_frame(dict_data):
    '''
    please provide dictionary as input 
    nj
    '''
    df_data = {}
    for key in dict_data.keys():
        df_data[key] = pd.DataFrame(dict_data.get(key)).T
    return df_data    



def schedule(data_dict,population_size = 30, crossover_rate = 0.8, \
             mutation_rate = 0.2, mutation_selection_rate = 0.2, num_iteration = 50):
    # initialization setting
#     pt_tmp = pd.read_excel("JSP_dataset.xlsx", sheet_name="Processing Time", index_col = [0])
#     ms_tmp = pd.read_excel("JSP_dataset.xlsx", sheet_name="Machines Sequence", index_col = [0])
    dfs  = from_json_to_frame(data_dict)
    ms_tmp = dfs['Machines Sequence']
    pt_tmp = dfs['Processing Time']

    dfshape = pt_tmp.shape
    num_mc = dfshape[1] # number of machines
    num_job = dfshape[0] # number of jobs
    num_gene = num_mc * num_job # number of genes in a chromosome

    pt = [list(map(int, pt_tmp.iloc[i])) for i in range(num_job)]
    ms = [list(map(int, ms_tmp.iloc[i])) for i in range(num_job)]


#     population_size = int(input('Please input the size of population: ') or 30) # default value is 30
#     crossover_rate = float(input('Please input the size of Crossover Rate: ') or 0.8) # default value is 0.8
#     mutation_rate = float(input('Please input the size of Mutation Rate: ') or 0.2) # default value is 0.2
#     mutation_selection_rate = float(input('Please input the mutation selection rate: ') or 0.2)
    num_mutation_jobs = round(num_gene * mutation_selection_rate)
#     num_iteration = int(input('Please input number of iteration: ') or 2000) # default value is 2000

    start_time = time.time()


    # generate initial population
    Tbest = 999999999999999
    best_list, best_obj = [], []
    population_list = []
    makespan_record = []
    for i in range(population_size):
        nxm_random_num = list(np.random.permutation(num_gene)) # generate a random permutation of 0 to num_job*num_mc-1
        population_list.append(nxm_random_num) # add to the population_list
        for j in range(num_gene):
            population_list[i][j] = population_list[i][j] % num_job # convert to job number format, every job appears m times

    for n in range(num_iteration):
        Tbest_now = 99999999999           

        # two point crossover 
        parent_list = copy.deepcopy(population_list)
        offspring_list = copy.deepcopy(population_list)
        S = list(np.random.permutation(population_size)) # generate a random sequence to select the parent chromosome to crossover

        for m in range(int(population_size/2)):
            crossover_prob = np.random.rand()
            if crossover_rate >= crossover_prob:
                parent_1 = population_list[S[2*m]][:]
                parent_2 = population_list[S[2*m+1]][:]
                child_1 = parent_1[:]
                child_2 = parent_2[:]
                cutpoint = list(np.random.choice(num_gene, 2, replace=False))
                cutpoint.sort()

                child_1[cutpoint[0]:cutpoint[1]] = parent_2[cutpoint[0]:cutpoint[1]]
                child_2[cutpoint[0]:cutpoint[1]] = parent_1[cutpoint[0]:cutpoint[1]]
                offspring_list[S[2*m]] = child_1[:]
                offspring_list[S[2*m+1]] = child_2[:]


        # repairment
        for m in range(population_size):
            job_count = {}
            larger, less=[], [] # 'larger' record jobs appear in the chromosome more than m times, and 'less' records less than m times.
            for i in range(num_job):
                if i in offspring_list[m]:
                    count = offspring_list[m].count(i)
                    pos = offspring_list[m].index(i)
                    job_count[i] = [count,pos] # store the above two values to the job_count dictionary
                else:
                    count = 0
                    job_count[i] = [count,0]
                if count > num_mc:
                    larger.append(i)
                elif count < num_mc:
                    less.append(i)

            for k in range(len(larger)):
                chg_job = larger[k]
                while job_count[chg_job][0] > num_mc:
                    for d in range(len(less)):
                        if job_count[less[d]][0] < num_mc:                    
                            offspring_list[m][job_count[chg_job][1]] = less[d]
                            job_count[chg_job][1] = offspring_list[m].index(chg_job)
                            job_count[chg_job][0] = job_count[chg_job][0]-1
                            job_count[less[d]][0] = job_count[less[d]][0]+1                    
                        if job_count[chg_job][0] == num_mc:
                            break     

        # mutatuon  
        for m in range(len(offspring_list)):
            mutation_prob = np.random.rand()
            if mutation_rate >= mutation_prob:
                m_chg = list(np.random.choice(num_gene, num_mutation_jobs, replace=False)) # chooses the position to mutation
                t_value_last = offspring_list[m][m_chg[0]] # save the value which is on the first mutation position
                for i in range(num_mutation_jobs-1):
                    offspring_list[m][m_chg[i]] = offspring_list[m][m_chg[i+1]] # displacement

                offspring_list[m][m_chg[num_mutation_jobs-1]]=t_value_last # move the value of the first mutation position to the last mutation position


        # fitness value(calculate makespan)
        total_chromosome=copy.deepcopy(parent_list)+copy.deepcopy(offspring_list) # parent and offspring chromosomes combination
        chrom_fitness,chrom_fit = [], []
        total_fitness = 0
        for m in range(population_size*2):
            j_keys = [j for j in range(num_job)]
            key_count = {key:0 for key in j_keys}
            j_count = {key:0 for key in j_keys}
            m_keys = [j+1 for j in range(num_mc)]
            m_count = {key:0 for key in m_keys}

            for i in total_chromosome[m]:
                gen_t = int(pt[i][key_count[i]])
                gen_m = int(ms[i][key_count[i]])
                j_count[i] = j_count[i]+gen_t
                m_count[gen_m] = m_count[gen_m]+gen_t

                if m_count[gen_m] < j_count[i]:
                    m_count[gen_m] = j_count[i]
                elif m_count[gen_m] > j_count[i]:
                    j_count[i] = m_count[gen_m]

                key_count[i] = key_count[i]+1

            makespan = max(j_count.values())
            chrom_fitness.append(1/makespan)
            chrom_fit.append(makespan)
            total_fitness = total_fitness+chrom_fitness[m]


        # selection(roulette wheel approach)
        pk, qk = [], []

        for i in range(population_size * 2):
            pk.append(chrom_fitness[i] / total_fitness)
        for i in range(population_size * 2):
            cumulative = 0
            for j in range(0, i+1):
                cumulative = cumulative + pk[j]
            qk.append(cumulative)

        selection_rand = [np.random.rand() for i in range(population_size)]

        for i in range(population_size):
            if selection_rand[i] <= qk[0]:
                population_list[i] = copy.deepcopy(total_chromosome[0])
            else:
                for j in range(0, population_size * 2-1):
                    if selection_rand[i] > qk[j] and selection_rand[i] <= qk[j+1]:
                        population_list[i] = copy.deepcopy(total_chromosome[j+1])
                        break
        # comparison
        for i in range(population_size * 2):
            if chrom_fit[i] < Tbest_now:
                Tbest_now = chrom_fit[i]
                sequence_now = copy.deepcopy(total_chromosome[i])
        if Tbest_now <= Tbest:
            Tbest = Tbest_now
            sequence_best = copy.deepcopy(sequence_now)

        makespan_record.append(Tbest)
    # result
    # print("optimal sequence", sequence_best)
    # print("optimal value:%f"%Tbest)
    # print('the elapsed time:%s'% (time.time() - start_time))



    # #%matplotlib inline
    # plt.plot([i for i in range(len(makespan_record))],makespan_record,'b')
    # plt.ylabel('makespan', fontsize=15)
    # plt.xlabel('generation', fontsize=15)
    # plt.show()


    # plot gantt chart

    m_keys = [j+1 for j in range(num_mc)]
    j_keys = [j for j in range(num_job)]
    key_count = {key:0 for key in j_keys}
    j_count = {key:0 for key in j_keys}
    m_count = {key:0 for key in m_keys}
    j_record = {}
    for i in sequence_best:
        gen_t = int(pt[i][key_count[i]])
        gen_m = int(ms[i][key_count[i]])
        j_count[i] = j_count[i]+gen_t
        m_count[gen_m] = m_count[gen_m]+gen_t

        if m_count[gen_m] < j_count[i]:
            m_count[gen_m] = j_count[i]
        elif m_count[gen_m] > j_count[i]:
            j_count[i] = m_count[gen_m]

        start_time = str(datetime.timedelta(seconds=j_count[i]-pt[i][key_count[i]])) # convert seconds to hours, minutes and seconds
        end_time = str(datetime.timedelta(seconds=j_count[i]))

        j_record[(i,gen_m)] = [start_time,end_time]

        key_count[i] = key_count[i]+1


    df = []
    for m in m_keys:
        for j in j_keys:
            try:
                df.append(dict(Task='Machine %s'%(m), name='Machine %s'%(m), Start='2020-02-01 %s'%(str(j_record[(j,m)][0])), Finish='2020-02-01 %s'%(str(j_record[(j,m)][1])),Resource='Job %s'%(j+1),  id='Machine_%s_'%(m)+'Job_%s'%(j+1)))
            except Exception as e:
                print((j,m))
                print(e)
            #print(df)
    
    df_ = pd.DataFrame(df)
    df_.Start = pd.to_datetime(df_['Start'])
    df_.Finish = pd.to_datetime(df_['Finish'])
    start = df_.Start.min()
    end = df_.Finish.max()

    df_.Start = df_.Start.apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S'))
    df_.Finish = df_.Finish.apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S'))
    data = df_.to_dict('record')

    final_data ={
        'start':start.strftime('%Y-%m-%dT%H:%M:%S'),
        'end':end.strftime('%Y-%m-%dT%H:%M:%S'),
        'data':data}
        
    # fig = ff.create_gantt(df, index_col='Resource', show_colorbar=True, group_tasks=True, showgrid_x=True, title='Job shop Schedule')
    # fig.show()
    # iplot(fig, filename='GA_job_shop_scheduling')
    return final_data







def setup_sch(data_json, population_size = 30, crossover_rate = 0.8, mutation_rate = 0.2, mutation_selection_rate = 0.5, num_iteration = 500):

    json_df = pd.DataFrame.from_dict(data_json)
    change = pd.DataFrame.from_dict(json_df['changeOverTime'][0])
    change = change.set_index('prod')
    p = json_df[['product', 'totalTime']].groupby(['product'], sort=False).sum()['totalTime'].to_list()
    json_df['dueDate'] = pd.to_datetime(json_df['dueDate'])
    json_df['project_start_date'] = pd.to_datetime(json_df['project_start'])
    json_df['due_date_int'] = (json_df['dueDate'] - json_df['project_start_date']).dt.days
    d = json_df[['product', 'due_date_int']].drop_duplicates()['due_date_int'].to_list()
    w = json_df[['product', 'Priority/Weights']].drop_duplicates()['Priority/Weights'].to_list()
    ch = change.values.tolist()
    num_job = len(p)


    #ch = [[0, 2, 3, 4, 5], [5,0,7,8, 6], [7, 8, 0, 2, 9], [7, 8, 1, 0, 9], [7, 8, 10, 2, 0]]


    # population_size=int(input('Please input the size of population: ') or 30) # default value is 30
    # crossover_rate=float(input('Please input the size of Crossover Rate: ') or 0.8) # default value is 0.8
    # mutation_rate=float(input('Please input the size of Mutation Rate: ') or 0.1) # default value is 0.1
    # mutation_selection_rate=float(input('Please input the mutation selection rate: ') or 0.5)
    num_mutation_jobs=round(num_job*mutation_selection_rate)
    # num_iteration=int(input('Please input number of iteration: ') or 500) # default value is 2000


    # start_time = time.time()

    '''==================== main code ==============================='''
    '''----- generate initial population -----'''
    Tbest=999999999999999
    best_list,best_obj=[],[]
    population_list=[]
    for i in range(population_size):
        random_num=list(np.random.permutation(num_job)) # generate a random permutation of 0 to num_job-1
        population_list.append(random_num) # add to the population_list
            
    for n in range(num_iteration):
        Tbest_now=99999999999           
        '''-------- crossover --------'''
        parent_list=copy.deepcopy(population_list)
        offspring_list=copy.deepcopy(population_list)
        S=list(np.random.permutation(population_size)) # generate a random sequence to select the parent chromosome to crossover
        
        for m in range(int(population_size/2)):
            crossover_prob=np.random.rand()
            if crossover_rate>=crossover_prob:
                parent_1= population_list[S[2*m]][:]
                parent_2= population_list[S[2*m+1]][:]
                child_1=['na' for i in range(num_job)]
                child_2=['na' for i in range(num_job)]
                fix_num=round(num_job/2)
                g_fix=list(np.random.choice(num_job, fix_num, replace=False))
                
                for g in range(fix_num):
                    child_1[g_fix[g]]=parent_2[g_fix[g]]
                    child_2[g_fix[g]]=parent_1[g_fix[g]]
                c1=[parent_1[i] for i in range(num_job) if parent_1[i] not in child_1]
                c2=[parent_2[i] for i in range(num_job) if parent_2[i] not in child_2]
                
                for i in range(num_job-fix_num):
                    child_1[child_1.index('na')]=c1[i]
                    child_2[child_2.index('na')]=c2[i]
                offspring_list[S[2*m]]=child_1[:]
                offspring_list[S[2*m+1]]=child_2[:]
            
        '''--------mutatuon--------'''   
        for m in range(len(offspring_list)):
            mutation_prob=np.random.rand()
            if mutation_rate >= mutation_prob:
                m_chg=list(np.random.choice(num_job, num_mutation_jobs, replace=False)) # chooses the position to mutation
                t_value_last=offspring_list[m][m_chg[0]] # save the value which is on the first mutation position
                for i in range(num_mutation_jobs-1):
                    offspring_list[m][m_chg[i]]=offspring_list[m][m_chg[i+1]] # displacement
                
                offspring_list[m][m_chg[num_mutation_jobs-1]]=t_value_last # move the value of the first mutation position to the last mutation position
        
        
        '''--------fitness value(calculate tardiness)-------------'''
        total_chromosome=copy.deepcopy(parent_list)+copy.deepcopy(offspring_list) # parent and offspring chromosomes combination
        chrom_fitness,chrom_fit=[],[]
        total_fitness=0
        for i in range(population_size*2):
            ptime=0
            tardiness=0
            change_list = []
            for j in range(num_job):
                if j == 0:
                    ptime=ptime+p[total_chromosome[i][j]]
                    tardiness=tardiness+w[total_chromosome[i][j]]*max(ptime-d[total_chromosome[i][j]],0)
                    change_list.append(total_chromosome[i][j])
                
                else:
                    ptime=ptime+p[total_chromosome[i][j]] + ch[total_chromosome[i][j]][change_list[j-1]]
                    tardiness=tardiness+w[total_chromosome[i][j]]*max(ptime-d[total_chromosome[i][j]],0)
                    change_list.append(total_chromosome[i][j])

                
            chrom_fitness.append(1/tardiness)
            chrom_fit.append(tardiness)
            total_fitness=total_fitness+chrom_fitness[i]
        
        '''----------selection----------'''
        pk,qk=[],[]
        
        for i in range(population_size*2):
            pk.append(chrom_fitness[i]/total_fitness)
        for i in range(population_size*2):
            cumulative=0
            for j in range(0,i+1):
                cumulative=cumulative+pk[j]
            qk.append(cumulative)
        
        selection_rand=[np.random.rand() for i in range(population_size)]
        
        for i in range(population_size):
            if selection_rand[i]<=qk[0]:
                population_list[i]=copy.deepcopy(total_chromosome[0])
            else:
                for j in range(0,population_size*2-1):
                    if selection_rand[i]>qk[j] and selection_rand[i]<=qk[j+1]:
                        population_list[i]=copy.deepcopy(total_chromosome[j+1])
                        break
        '''----------comparison----------'''
        for i in range(population_size*2):
            if chrom_fit[i]<Tbest_now:
                Tbest_now=chrom_fit[i]
                sequence_now=copy.deepcopy(total_chromosome[i])
        
        if Tbest_now<=Tbest:
            Tbest=Tbest_now
            sequence_best=copy.deepcopy(sequence_now)
        
        job_sequence_ptime=0
        num_tardy=0
        for k in range(num_job):
            job_sequence_ptime=job_sequence_ptime+p[sequence_best[k]]
            if job_sequence_ptime>d[sequence_best[k]]:
                num_tardy=num_tardy+1
    '''----------result----------'''
    # print("optimal sequence",sequence_best)
    # print("optimal value:%f"%Tbest)
    # print("average tardiness:%f"%(Tbest/num_job))
    # print("number of tardy:%d"%num_tardy)
    # print('the elapsed time:%s'% (time.time() - start_time))

    #'''--------plot gantt chart-------'''

    j_keys=[j for j in range(num_job)]
    j_count={key:0 for key in j_keys}
    m_count=0
    j_record={}
    count = 0
    check_lis = []
    change_over_time =[]
    for i in sequence_best:
        if count == 0:
            check_lis.append(i)
            change_over_time.append(0)
        else :
            check_lis.append(i)
            change_over_time.append((ch[check_lis[count-1]][i]))
        count += 1

    change_over_add = []
    sum = 0
    for i in change_over_time:
        sum += i
        change_over_add += [sum]

    count_change = 0
    for i in sequence_best:
        gen_t=int(p[i])
        j_count[i]=j_count[i]+gen_t
        m_count=m_count+gen_t
        if m_count<j_count[i]:
            m_count=j_count[i]
        elif m_count>j_count[i]:
            j_count[i]=m_count

        
        # adding static value for date that need to be replaced with project start date
        date_str = json_df['project_start'][0]
        date_dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')

        start_time=str(date_dt + datetime.timedelta(seconds=(j_count[i]-p[i] + change_over_add[count_change])*60)) # convert time
        end_time=str(date_dt + datetime.timedelta(seconds=(j_count[i] + change_over_add[count_change])*60))

        #print(start_time, j_count[i]-p[i] + change_over_add[count_change], end_time)

        count_change += 1

    
        j_record[i]=[start_time,end_time]


    df = []
    for j in j_keys:
        df.append(dict(name='Machine -1', Start='%s'%(str(j_record[j][0])), Finish='%s'%(str(j_record[j][1])),product_id='Product - %s'%(j+1), id='%s'%(j+1)))

    # colors={}
    # for i in j_keys:
    #     colors['Job %s'%(i+1)]='rgb(%s,%s,%s)'%(255/(i+1)+0*i,5+12*i,50+10*i)
    # fig = ff.create_gantt(df, colors=['#008B00','#FF8C00','#E3CF57','#0000CD','#7AC5CD','#ED9121','#76EE00','#6495ED','#008B8B','#A9A9A9','#A2CD5A','#9A32CD','#8FBC8F','#EEC900','#EEE685','#CDC1C5','#9AC0CD','#EEA2AD','#00FA9A','#CDB38B'], index_col='Resource', show_colorbar=True, group_tasks=True, showgrid_x=True)
    # fig.show()

                
    df_ = pd.DataFrame(df)
    df_.Start = pd.to_datetime(df_['Start'])
    df_.Finish = pd.to_datetime(df_['Finish'])
    start = df_.Start.min()
    end = df_.Finish.max()

    df_.Start = df_.Start.apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S'))
    df_.Finish = df_.Finish.apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S'))
    data_out = df_.to_dict('record')

    df_['Start'] =  pd.to_datetime(df_['Start'])


    data_df = json_df[['name', 'product', 'Task', 'id', 'resource', 'taskId', 'customer', 'salesOrder', 'dueDate', 'quantity', 'timeTake', 'Freeze', 'overlapEvent', 'totalTime', 'changeOverTime', 'Priority/Weights']]
    data_df['Freeze'] = False
    data_df['overlapEvent'] = False
    pi = list(df_['Start'])

    p = pd.concat([data_df, pd.DataFrame(columns = [ 'Start', 'Finish'])])
    prod = list(p['product'].unique())

    p['Start'] =  pd.to_datetime(p['Start'], format='%d%b%Y:%H:%M:%S.%f')
    p['Finish'] =  pd.to_datetime(p['Finish'], format='%d%b%Y:%H:%M:%S.%f')

    for i in range(len(prod)):
        p.loc[(p['Task'] == p['Task'][0]) & (p['product'] == prod[i]),'Start'] = pi[i]


    for index, row in p.iterrows():
        p['Finish'][index] = p['Start'][index] + datetime.timedelta(seconds=(p['totalTime'][index])*60)
        if index < p.shape[0]-1:
            if p['Task'][index+1] != p['Task'][0]:
                p['Start'][index+1] = p['Finish'][index]


    #ADDED FOR SAME RESOURCE
    p.sort_values(by=['Start'],inplace=True)
    sorted_products = p['product'].unique()
    p.reset_index(inplace = True)

    for index,row in p.iterrows():
        if p['product'][index] != sorted_products[0]:
            p['Start'][index] = p['Finish'][index-5] + datetime.timedelta(seconds=change_over_time[(int(index/5))]*60)
            p['Finish'][index] = p['Start'][index] + datetime.timedelta(seconds=p['totalTime'][index]*60)

    p.sort_values(by=['index'],inplace=True)
    p.set_index(['index'], inplace = True)
    p.Start = p.Start.apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S'))
    p.Finish = p.Finish.apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S'))

    setup_sch_data = p.to_dict('record')

    final_data ={
        'start':start.strftime('%Y-%m-%dT%H:%M:%S'),
        'end':end.strftime('%Y-%m-%dT%H:%M:%S'),
        'data':setup_sch_data}

    return setup_sch_data







@api_view(['POST'])
def sc_view(request):
    data_dict = request.data['data_dict']
    # print(type(data_dict))
    # print(data_dict)
    final_data = schedule(data_dict)
    return Response(final_data)


@api_view(['GET'])
def sc_data(request):
    #data = from_excel_to_json(BASE_DIR + '/data/JSP_dataset.xlsx')
    # data = data_json.data_json
    return Response(data_json)


@api_view(['GET'])
def static_json(request):
    data = static_data(BASE_DIR + '/data/fjsp_product.json')
    # data = data_json.data_json
    return Response(data)



@api_view(['POST'])
def fjsp_view(request):
    data_dict = request.data
    # print(data_dict)
    final_data = setup_sch(data_dict)
    return Response(final_data)









""" production planning cadila """ 



def num_batch_campaign(data, batch_size, batch_changeover, duration_granulation_batch):
    data['Batches'] = np.ceil(data['Qty (kgs)']/batch_size)
    data['Campaigns'] = np.ceil(data['Batches']/10)
    data['Process_time_granulation'] = data['Batches']*duration_granulation_batch + data['Campaigns']*batch_changeover

    return data




def schedule_granulation(json_data, population_size=30, crossover_rate=0.8, mutation_rate=0.1, mutation_selection_rate=0.5, num_iteration=500):
    # INITIAL DATA
    # demand = pd.read_excel('Cadila_Design.xlsx', sheet_name='Demand')
    # #routing = pd.read_excel('Cadila_Design.xlsx', sheet_name='Routing')
    # change_over = pd.read_excel('Cadila_Design.xlsx', sheet_name='ChangeOver')

    demand_data = pd.DataFrame.from_dict(json_data['demand_unconstrained'])[['id', 'Products', 'Order Qty(pcs)', 'Qty (kgs)', 'Due date']]

    change_over = pd.DataFrame.from_dict(json_data['change_over'])

    #added changeover in a list for scheduling granulation batches first
    granulation_change_over = change_over.loc[change_over['Process'] == 'Granulation']['Time (Mins)'].to_list()

    """ ALL INPUTS for MODELING """

    batch_size_granulation = 684        # in kgs
    duration_granulation_batch = 456    # in minutes find from batch size devided by rate
    batch_to_batch_changeover = granulation_change_over[0]  # in miunutes


    demand = num_batch_campaign(demand_data, batch_size_granulation, batch_to_batch_changeover, duration_granulation_batch)

    p = demand['Process_time_granulation'].to_list()
    d = [20,20,20,20,20,20,20]   # add start date to find number of days by substracting from due date
    w = [1,1,1,1,1,1,1]

    num_job = len(p)


    num_mutation_jobs=round(num_job*mutation_selection_rate)



    '''==================== main code ==============================='''
    '''----- generate initial population -----'''
    Tbest=999999999999999
    best_list,best_obj=[],[]
    population_list=[]
    for i in range(population_size):
        random_num=list(np.random.permutation(num_job)) # generate a random permutation of 0 to num_job-1
        population_list.append(random_num) # add to the population_list
        
    for n in range(num_iteration):
        Tbest_now=99999999999           
        '''-------- crossover --------'''
        parent_list=copy.deepcopy(population_list)
        offspring_list=copy.deepcopy(population_list)
        S=list(np.random.permutation(population_size)) # generate a random sequence to select the parent chromosome to crossover
    
        for m in range(int(population_size/2)):
            crossover_prob=np.random.rand()
            if crossover_rate>=crossover_prob:
                parent_1= population_list[S[2*m]][:]
                parent_2= population_list[S[2*m+1]][:]
                child_1=['na' for i in range(num_job)]
                child_2=['na' for i in range(num_job)]
                fix_num=round(num_job/2)
                g_fix=list(np.random.choice(num_job, fix_num, replace=False))
            
                for g in range(fix_num):
                    child_1[g_fix[g]]=parent_2[g_fix[g]]
                    child_2[g_fix[g]]=parent_1[g_fix[g]]
                c1=[parent_1[i] for i in range(num_job) if parent_1[i] not in child_1]
                c2=[parent_2[i] for i in range(num_job) if parent_2[i] not in child_2]
            
                for i in range(num_job-fix_num):
                    child_1[child_1.index('na')]=c1[i]
                    child_2[child_2.index('na')]=c2[i]
                offspring_list[S[2*m]]=child_1[:]
                offspring_list[S[2*m+1]]=child_2[:]



            '''--------mutatuon--------'''   
        for m in range(len(offspring_list)):
            mutation_prob=np.random.rand()
            if mutation_rate >= mutation_prob:
                m_chg=list(np.random.choice(num_job, num_mutation_jobs, replace=False)) # chooses the position to mutation
                t_value_last=offspring_list[m][m_chg[0]] # save the value which is on the first mutation position
                for i in range(num_mutation_jobs-1):
                    offspring_list[m][m_chg[i]]=offspring_list[m][m_chg[i+1]] # displacement
            
                offspring_list[m][m_chg[num_mutation_jobs-1]]=t_value_last # move the value of the first mutation position to the last mutation position
                
        
        '''--------fitness value(calculate tardiness)-------------'''
        total_chromosome=copy.deepcopy(parent_list)+copy.deepcopy(offspring_list) # parent and offspring chromosomes combination
        chrom_fitness,chrom_fit=[],[]
        total_fitness=0
        for i in range(population_size*2):
            ptime=0
            tardiness=0
            for j in range(num_job):
                ptime=ptime+p[total_chromosome[i][j]]
                tardiness=tardiness+w[total_chromosome[i][j]]*max(ptime-d[total_chromosome[i][j]],0)
            chrom_fitness.append(1/tardiness)
            chrom_fit.append(tardiness)
            total_fitness=total_fitness+chrom_fitness[i]
            
            '''----------selection----------'''
        pk,qk=[],[]
        
        for i in range(population_size*2):
            pk.append(chrom_fitness[i]/total_fitness)
        for i in range(population_size*2):
            cumulative=0
            for j in range(0,i+1):
                cumulative=cumulative+pk[j]
            qk.append(cumulative)
    
        selection_rand=[np.random.rand() for i in range(population_size)]
    
        for i in range(population_size):
            if selection_rand[i]<=qk[0]:
                population_list[i]=copy.deepcopy(total_chromosome[0])
            else:
                for j in range(0,population_size*2-1):
                    if selection_rand[i]>qk[j] and selection_rand[i]<=qk[j+1]:
                        population_list[i]=copy.deepcopy(total_chromosome[j+1])
                        break
        '''----------comparison----------'''
        for i in range(population_size*2):
            if chrom_fit[i]<Tbest_now:
                Tbest_now=chrom_fit[i]
                sequence_now=copy.deepcopy(total_chromosome[i])
    
        if Tbest_now<=Tbest:
            Tbest=Tbest_now
            sequence_best=copy.deepcopy(sequence_now)
    
        job_sequence_ptime=0
        num_tardy=0
        for k in range(num_job):
            job_sequence_ptime=job_sequence_ptime+p[sequence_best[k]]
            if job_sequence_ptime>d[sequence_best[k]]:
                num_tardy=num_tardy+1


        
# '''----------result----------'''
# print("optimal sequence",sequence_best)
# print("optimal value:%f"%Tbest)
# print("average tardiness:%f"%(Tbest/num_job))
# print("number of tardy:%d"%num_tardy)


# #'''--------plot gantt chart-------'''
# import pandas as pd
# #import plotly.plotly as py
# #import plotly.figure_factory as ff
# #import plotly.offline as offline
# import datetime

    j_keys=[j for j in range(num_job)]
    j_count={key:0 for key in j_keys}
    m_count=0
    j_record={}
    for i in sequence_best:
        gen_t=int(p[i])
        j_count[i]=j_count[i]+gen_t
        m_count=m_count+gen_t
   
        if m_count<j_count[i]:
            m_count=j_count[i]
        elif m_count>j_count[i]:
            j_count[i]=m_count


    # adding static value for date that need to be replaced with project start date
        date_str = "2020-04-15"
        date_dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')

        start_time=str(date_dt + datetime.timedelta(seconds=(j_count[i]-p[i])*60)) # convert seconds to hours, minutes and seconds

        end_time=str(date_dt + datetime.timedelta(seconds=(j_count[i])*60))

        j_record[i]=[start_time,end_time]
       

    df=[]
    for j in j_keys:
        df.append(dict(Task='Machine', Start='%s'%(str(j_record[j][0])), Finish='%s'%(str(j_record[j][1])),Resource='Job %s'%(j+1), Product_id='10%s'%(j+1)))

    output_df = pd.DataFrame.from_dict(df)
    output_df



    output_df['Start'] = pd.to_datetime(output_df['Start'])
    output_df['Finish'] = pd.to_datetime(output_df['Finish'])
    output_df['totalTime'] = output_df['Finish']- output_df['Start']



    output_df.sort_values(by='Start', inplace=True)
    output_df.reset_index(drop=True,inplace=True)



    finish_g1 = []
    finish_g2 = []
    for index, row in output_df.iterrows():
        if index == 0:
            output_df['Task'][index] = 'G1'
            finish_g1.append(output_df['Finish'][index])
        elif index == 1:
            output_df['Task'][index] = 'G2'
            finish_g2.append(output_df['Finish'][index])
        else:
            if max(finish_g1)>=max(finish_g2):
                finish_g2.append(output_df['Finish'][index])
                output_df['Task'][index] = 'G2'

            else:
                finish_g1.append(output_df['Finish'][index])
                output_df['Task'][index] = 'G1'



    demand_campaign = num_batch_campaign(demand, batch_size_granulation, batch_to_batch_changeover, duration_granulation_batch)

    output_df['Product_id'] = output_df['Product_id'].astype(np.int64)
    result_df = pd.merge(output_df, demand_campaign, left_on='Product_id', right_on='id')
    result_df.drop('Product_id', axis=1, inplace=True)


    task_df = result_df[['Task', 'id', 'Products',
        'Order Qty(pcs)', 'Qty (kgs)', 'Due date', 'Batches', 'Campaigns',
        'Process_time_granulation']]

    task_df.rename(columns={'Order Qty(pcs)':'Order_Qty','Qty (kgs)':'Qty'}, inplace=True)
    task_df['Batches'] = task_df['Batches'].astype(int)

   
    return json_data, task_df




def input(filename,data):

    num_of_granulation_machine=pd.DataFrame.from_dict([{'Granulation': 'G1', 'Rate': 1.5, 'Output_Kgs': 684},
 {'Granulation': 'G2', 'Rate': 1.5, 'Output_Kgs': 684}])
    num_of_granulation_machine.rename(columns={'Output (Kgs)':'Output_Kgs'}, inplace=True)
    num_of_shift = pd.DataFrame.from_dict(filename['shifts'])
    col_name=['id','Task','Products','Batches','Qty','Qty_pcs','assigned_priority','shift']
    df=pd.DataFrame(columns=col_name)
    # x_G1=datetime.now()
    # x_G2=x_G1
    shift=0
    row_number=0
    for i in range(0,data.shape[0]):
        for j in range(0,data.Batches[i]):
            shift=shift+1
            row_value = [data.id[i],data.Task[i],data.Products[i],j+1,data.Qty[i],data.Order_Qty[i],row_number+1,shift]
    #         if(data.Task[i]=='G1'):
    #             y_G1=x_G1+timedelta(minutes=456)
    #             row_value = [data.id[i],data.Task[i],data.Products[i],j+1,output(data.Qty[i]),0,data.Order_Qty[i],row_number+1,shift,x_G1,y_G1]
    #             x_G1=y_G1
    #         else:
    #             y_G2=x_G2+timedelta(minutes=456)
    #             row_value = [data.id[i],data.Task[i],data.Products[i],j+1,output(data.Qty[i]),0,data.Order_Qty[i],row_number+1,shift,x_G2,y_G2]
    #             x_G2=y_G2
            if row_number > df.index.max()+1: 
                print("Invalid row_number") 
            else:
                df = Insert_row(row_number, df, row_value) 
            row_number=row_number+1
            #li.append(df.Task[j])
            # if(shift==num_of_shift.shape[0]):
            #     shift=0
            #print(df.Qty_kgs[row_number-1],end=' ')
    #         df.at[row_number-1,'Time']=df.Qty_kgs[row_number-1]/rate_G1
    #     if(df.Qty_kgs[row_number-1]>=output_G1):
    #         #print("YES",end=" ")
    #         df.at[row_number-1,'Qty_kgs']=data.Qty[i]-(data.Batches[i]-1)*684
    #       df.at[row_number-1,'Time']=df.Qty_kgs[row_number-1]/rate_G1
    #     if()
    #         df.end[row_number-1]=df.start[row_number-1]+timedelta(minutes=df.Time[row_number-1])
    #     else:
    #         df.end[row_number-1]=df.start[row_number-1]+timedelta(minutes=df.Time[row_number-1])

    dict_granulation=num_of_granulation_machine.to_dict()
    shift=0
    for i in range(0,num_of_granulation_machine.shape[0]):
        machine=dict_granulation['Granulation'][i]
        for j in range(0,df.shape[0]):
            if(machine==df.Task[j]):
                shift=shift+1
                df['shift'][j]=shift
                if(shift==num_of_shift.shape[0]):
                    shift=0
        shift=0

    return granulation(filename,df,data)



def Insert_row(row_number, df, row_value): 
    start_upper = 0
    end_upper = row_number 
    start_lower = row_number 
    end_lower = df.shape[0] 
    upper_half = [*range(start_upper, end_upper, 1)] 
    lower_half = [*range(start_lower, end_lower, 1)] 
    lower_half = [x.__add__(1) for x in lower_half]
    index_ = upper_half + lower_half 
    df.index = index_ 
    df.loc[row_number] = row_value 
    df = df.sort_index() 
    return df



def update(machine,diff,pos,df):
    for j in range(pos,df.shape[0]):
        if(df['Task'][j]==machine):
            #print(j,'j : ')
            df['Start_Granulation'][j]=diff+df['Start_Granulation'][j]
            df['End_Granulation'][j]=diff+df['End_Granulation'][j]
    return df


def granulation(filename,df,data):
    #dispensing=pd.DataFrame([{'Dispensing Time': 30}])
    dispensing = pd.DataFrame.from_dict(filename['process_change_over'])
    num_of_compression_machine=pd.DataFrame.from_dict(filename['compression_capacity'])
    num_of_compression_machine.rename(columns={'Output (Kgs)':'Output_Kgs','Rate(in min)':'Rate'}, inplace=True)
    changeover_time=pd.DataFrame.from_dict(filename['change_over'])
    num_of_shift = pd.DataFrame.from_dict(filename['shifts'])
    num_of_granulation_machine=pd.DataFrame.from_dict([{'Granulation': 'G1', 'Rate': 1.5, 'Output_Kgs': 684},
 {'Granulation': 'G2', 'Rate': 1.5, 'Output_Kgs': 684}])
    num_of_granulation_machine.rename(columns={'Output (Kgs)':'Output_Kgs'}, inplace=True)

    dict_granulation=num_of_granulation_machine.to_dict()
#    dict_granulation

    df['Qty_Kgs']=np.zeros(df.shape[0])
    a=datetime.datetime(2020,5,6,12,0,0,0)
    num_machine=num_of_granulation_machine.shape[0]
    #dispense_time=dispensing['Dispensing Time'][0]
    dispense_time=dispensing.loc[dispensing['Process-Process Changeover']=='Dispensing Time']['Time(Mins)'][0]
    temp=0
    df['Start_Granulation']=np.zeros(df.shape[0])
    df['Time_Granulation']=np.zeros(df.shape[0])
    df['End_Granulation']=np.zeros(df.shape[0])
    for i in range(0,changeover_time.shape[0]):
        if(changeover_time.Process[i]=='Granulation'):
            granulation_change=changeover_time['Time (Mins)'][i]
            product_product=changeover_time['Time (Mins)'][i+1]
            break
    for i in range(0,data.shape[0]):
        for j in range(0,num_machine):
            if(dict_granulation['Granulation'][j]==data.Task[i]):
                break
        if(dict_granulation['Output_Kgs'][j]<data.Qty[i]):
            #print(temp)
            output=dict_granulation['Output_Kgs'][j]
            df.Qty_Kgs[temp:(temp+data.Batches[i]+1)]=output
            df.Time_Granulation[temp:(temp+data.Batches[i]+1)]=output/dict_granulation['Rate'][j]
            df.Qty_Kgs[temp+data.Batches[i]-1]=data.Qty[i]-(data.Batches[i]-1)*(dict_granulation['Output_Kgs'][j])
            df.Time_Granulation[temp+data.Batches[i]-1]=(data.Qty[i]-((data.Batches[i]-1)*(output)))/dict_granulation['Rate'][j]
            temp=temp+data.Batches[i]
        else:
            df.Qty_Kgs[temp]=data.Qty[i]
            df.Time_Granulation[temp]=data.Qty[i]/dict_granulation['Rate'][j]
            temp=temp+data.Batches[i]
            #df.at[temp-1,'Qty_kgs']=1
    # temp=0
    # for i in range(0,data.shape[0]):
    #     df.Start_Granulation[temp]=df.Start_Granulation[temp]+timedelta(minutes=int(dispense_time))
    #     df.End_Granulation[temp]=df.End_Granulation[temp]+timedelta(minutes=int(dispense_time))
    #     if(i>0):
    #         df.Start_Granulation[temp]=df.Start_Granulation[temp]+timedelta(minutes=int(product_product))
    #         df.End_Granulation[temp]=df.End_Granulation[temp]+timedelta(minutes=int(product_product))
    #     temp=temp+data.Batches[i]
    k=0
    for j in range(0,num_machine):
        machine=dict_granulation['Granulation'][j]
        for i in range(0,df.shape[0]):
            if(df.Task[i]==machine):
                #print(i,k)
                if(i-k!=1):
                    df.Start_Granulation[i]=dispense_time+product_product
                    df.End_Granulation[i]=dispense_time+product_product
                k=i
    df.Start_Granulation[0]=df.Start_Granulation[0]-product_product
    df.End_Granulation[0]=df.End_Granulation[1]-product_product
    start=datetime.datetime.strptime(filename['projectStart'], '%Y-%m-%d')+datetime.timedelta(hours=6)
    shift1=0
    for j in range(0,num_machine):
        machine=dict_granulation['Granulation'][j]
        for i in range(0,df.shape[0]):
            if(df.Task[i]==machine):
                shift1=shift1+1
                df['shift'][i]=shift1
                if(shift1==num_of_shift.shape[0]):
                    shift1=0
                df['Start_Granulation'][i]=start+datetime.timedelta(minutes=int(df['Start_Granulation'][i]))
                df['End_Granulation'][i]=df['Start_Granulation'][i]+datetime.timedelta(minutes=df.Time_Granulation[i])
                #start=start+timedelta(minutes=df.Time_Granulation[i]+granulation_change)
    #             if(temp==i):
    #                 print(i,temp)
    #                 df.Start_Granulation[temp]=df.Start_Granulation[temp]+timedelta(minutes=int(dispense_time))
    #                 df.End_Granulation[temp]=df.End_Granulation[temp]+timedelta(minutes=int(dispense_time))    
    #                 #start=df.End_Granulation[temp]
    #                 temp=temp+data.Batches[k]
    #                 print(temp)
    #                 k=k+1
                start=df['End_Granulation'][i]+datetime.timedelta(minutes=int(granulation_change))
        start=datetime.datetime.strptime(filename['projectStart'], '%Y-%m-%d')+datetime.timedelta(hours=6)
        shift1=0



    #time_dict = json.loads(filename)
    if "time" in filename:
        print('in time dict')

        time = pd.DataFrame.from_dict(filename['time'])
        time['fTime'] = time['date']+time['fTime']
        time['tTime'] = time['date']+time['tTime']
        # Check if granulation exist
        if 'Granulation I' in time.values:
            time.loc[time['machine'] == 'Granulation I', 'machine'] = 'G1'
        if 'Granulation II' in time.values:
            time.loc[time['machine'] == 'Granulation II','machine'] = 'G2'
        time['date'] = pd.to_datetime(time['date'])

        dict_time=time.to_dict()
        for j in range(0,time.shape[0]):
            time['fTime'][j] = datetime.datetime.strptime(time['fTime'][j], '%Y-%m-%d%H:%M')
            time['tTime'][j] = datetime.datetime.strptime(time['tTime'][j], '%Y-%m-%d%H:%M')
        for j in range(0,time.shape[0]):
            machine=time['machine'][j]
            for i in range(0,df.shape[0]):
                if(machine==df['Task'][i]):
                    if(df['Start_Granulation'][i].date()==time['date'][j].date() or df['End_Granulation'][i].date()==time['date'][j].date()):
                        if(df['Start_Granulation'][i].date()==time['date'][j].date()):
                            if(df['Start_Granulation'][i]>time['fTime'][j] and df['End_Granulation'][i]<time['tTime'][j]):
                                #print('up',i)
                                diff=time['tTime'][j]-df['Start_Granulation'][i]
                                df=update(machine,diff,i,df)
                        if(df['End_Granulation'][i].date()==time['date'][j].date()):
                            if(time['fTime'][j]<df['End_Granulation'][i]):
                                #print(i)
                                #print(df['Start_Granulation'][i],time['tTime'][j])
                                diff=time['tTime'][j]-df['Start_Granulation'][i]
                                #print(diff)
                                df=update(machine,diff,i,df)


    df.sort_values("End_Granulation", axis = 0, ascending = True, 
                     inplace = True, na_position ='last')
    li1=[]
    i=0
    while(i<df.shape[0]):
        for j in range(0,num_of_compression_machine.shape[0]):
            if i < df.shape[0]:
                li1.append(num_of_compression_machine.Compression[j])
            i=i+1


    df['Compression']=li1
    df.sort_values("assigned_priority", axis = 0, ascending = True, 
                 inplace = True, na_position ='last') 
    

    return compression(filename,df)
        

def update1(machine,diff,pos,df):
    for j in range(pos,df.shape[0]):
        if(df['Compression'][j]==machine):
            print(df['Start_Compression'][j],df['End_Compression'][j],diff)
            df['Start_Compression'][j]=diff+df['Start_Compression'][j]
            print(df['Start_Compression'][j],df['End_Compression'][j])
            df['End_Compression'][j]=diff+df['End_Compression'][j]
            print(df['Start_Compression'][j],df['End_Compression'][j])
    return df


def compression(filename,df):
    num_of_compression_machine=pd.DataFrame.from_dict(filename['compression_capacity'])
    num_of_compression_machine.rename(columns={'Output (Kgs)':'Output_Kgs','Rate(in min)':'Rate'}, inplace=True)
    num_of_coating_machine=pd.DataFrame.from_dict(filename['coating_capacity'])
    num_of_coating_machine.rename(columns={'Output (Kgs)':'Output_Kgs'}, inplace=True)
    changeover_time=pd.DataFrame.from_dict(filename['change_over'])
    num_of_granulation_machine=pd.DataFrame.from_dict([{'Granulation': 'G1', 'Rate': 1.5, 'Output_Kgs': 684},
 {'Granulation': 'G2', 'Rate': 1.5, 'Output_Kgs': 684}])
    num_of_granulation_machine.rename(columns={'Output (Kgs)':'Output_Kgs'}, inplace=True)
    dict_granulation=num_of_granulation_machine.to_dict()
    dict_compression=num_of_compression_machine.to_dict()
    for i in range(0,changeover_time.shape[0]):
        if(changeover_time.Process[i]=='Compression'):
            compression_change=changeover_time['Time (Mins)'][i]
            product_product=changeover_time['Time (Mins)'][i+1]
            break
    num_machine=num_of_compression_machine.shape[0]
    temp=0
    df['Start_Compression']=np.zeros(df.shape[0])
    df['Time_Compression']=np.zeros(df.shape[0])
    df['End_Compression']=np.zeros(df.shape[0])
    for i in range(0,df.shape[0]):
        for j in range(0,num_machine):
            if(dict_compression['Compression'][j]==df.Compression[i]):
                 break
        df['Time_Compression'][i]=df.Qty_pcs[i]/dict_compression['Rate'][j]
        k=0
    for j in range(0,num_machine):
        machine=dict_granulation['Granulation'][j]
        for i in range(0,df.shape[0]):
            if(df.Task[i]==machine):
                #print(i,k)
                if(i-k!=1):
                    df.Start_Compression[i]=product_product
                    df.End_Compression[i]=product_product
                k=i
    df.Start_Compression[0]=df.Start_Compression[0]-product_product
    df.End_Compression[0]=df.End_Compression[1]-product_product
    

    process_change = pd.DataFrame.from_dict(filename['process_change_over'])
    process = process_change['Time(Mins)'][1]
    for j in range(0,num_machine):
        machine=dict_compression['Compression'][j]
        l=0
        for i in range(0,df.shape[0]):
            if(df.Compression[i]==machine):
               # print(df.Start_Compression[i])
                df['Start_Compression'][i]=datetime.timedelta(minutes=int(process))+df.End_Granulation[i]
                df['End_Compression'][i]=df['Start_Compression'][i]+datetime.timedelta(minutes=int(df.Time_Compression[i]))


                if(i>0 and df['End_Compression'][l]!=0.0 and df['Start_Compression'][i]<df['End_Compression'][l]):
                    delay=df['End_Compression'][l]-df['Start_Compression'][i]
                    df['Start_Compression'][i]=df['Start_Compression'][i]+(delay)+datetime.timedelta(minutes=int(process))
                    df['End_Compression'][i]=df['End_Compression'][i]+(delay)+datetime.timedelta(minutes=int(process))
                l=i


    if "time" in filename:
        time = pd.DataFrame.from_dict(filename['time'])
        time['fTime'] = time['date']+time['fTime']
        time['tTime'] = time['date']+time['tTime']

        time['date'] = pd.to_datetime(time['date'])

        for j in range(0,time.shape[0]):
            time['fTime'][j] = datetime.datetime.strptime(time['fTime'][j], '%Y-%m-%d%H:%M')
            time['tTime'][j] = datetime.datetime.strptime(time['tTime'][j], '%Y-%m-%d%H:%M')

        for j in range(0,time.shape[0]):
            machine=time['machine'][j]
            for i in range(0,df.shape[0]):
                if(machine==df['Compression'][i]):
                    #print(i,machine)
                    if(df['Start_Compression'][i].date()==time['date'][j].date() or df['End_Compression'][i].date()==time['date'][j].date()):
                        if(df['Start_Compression'][i].date()==time['date'][j].date()):
                            if(df['Start_Compression'][i]>time['fTime'][j] and df['End_Compression'][i]<time['tTime'][j]):
                                print('up',i)
                                diff=time['tTime'][j]-df['Start_Compression'][i]
                                df=update1(machine,diff,i,df)
                        if(df['Start_Compression'][i].date()!=time['date'][j].date() and df['End_Compression'][i].date()==time['date'][j].date()):
                            if(time['fTime'][j]<df['End_Compression'][i]):
                                print(i)
                                #print(df['Start_Granulation'][i],time['tTime'][j])
                                diff=time['tTime'][j]-df['Start_Compression'][i]
                                #print(diff)
                                df=update1(machine,diff,i,df)


    df.sort_values("End_Compression", axis = 0, ascending = True, 
                 inplace = True, na_position ='last') 
    li1=[]
    i=0
    while(i<df.shape[0]):
        for j in range(0,num_of_coating_machine.shape[0]):
            if i < df.shape[0]:
                li1.append(num_of_coating_machine.Coating[j])
            i=i+1
    df['Coating']=li1
    df.sort_values("assigned_priority", axis = 0, ascending = True, 
                 inplace = True, na_position ='last') 
    return coating(filename,df)
    


def update2(machine,diff,pos,df):
    for j in range(pos,df.shape[0]):
        if(df['Coating'][j]==machine):
            #print(df['Start_Compression'][j],df['End_Compression'][j],diff)
            df['Start_Coating'][j]=diff+df['Start_Coating'][j]
            #print(df['Start_Compression'][j],df['End_Compression'][j])
            df['End_Coating'][j]=diff+df['End_Coating'][j]
            #print(df['Start_Compression'][j],df['End_Compression'][j])
    return df



def coating(filename,df):
    num_of_inspection_machine=pd.DataFrame.from_dict(filename['inspection_capacity'])

    num_of_coating_machine=pd.DataFrame.from_dict(filename['coating_capacity'])
    num_of_coating_machine.rename(columns={'Output (Kgs)':'Output_Kgs'}, inplace=True)
    changeover_time=pd.DataFrame.from_dict(filename['change_over'])
    num_of_granulation_machine=pd.DataFrame.from_dict([{'Granulation': 'G1', 'Rate': 1.5, 'Output_Kgs': 684},
 {'Granulation': 'G2', 'Rate': 1.5, 'Output_Kgs': 684}])
    num_of_granulation_machine.rename(columns={'Output (Kgs)':'Output_Kgs'}, inplace=True)
    dict_granulation=num_of_granulation_machine.to_dict()
    dict_coating=num_of_coating_machine.to_dict()
    for i in range(0,changeover_time.shape[0]):
        if(changeover_time.Process[i]=='Coating'):
            coating_change=changeover_time['Time (Mins)'][i]
            product_product=changeover_time['Time (Mins)'][i+1]
            break
    num_machine=num_of_coating_machine.shape[0]
    temp=0
    df['Start_Coating']=np.zeros(df.shape[0])
    df['Time_Coating']=np.zeros(df.shape[0])
    df['End_Coating']=np.zeros(df.shape[0])
    for i in range(0,df.shape[0]):
        for j in range(0,num_machine):
            if(dict_coating['Coating'][j]==df.Coating[i]):
                 break
        df['Time_Coating'][i]=df.Qty_Kgs[i]/dict_coating['Rate'][j]
    k=0
    for j in range(0,num_machine):
        machine=dict_granulation['Granulation'][j]
        for i in range(0,df.shape[0]):
            if(df.Task[i]==machine):
                #print(i,k)
                if(i-k!=1):
                    df.Start_Coating[i]=product_product
                    df.End_Coating[i]=product_product
                k=i
    df.Start_Coating[0]=df.Start_Coating[0]-product_product
    df.End_Coating[0]=df.End_Coating[1]-product_product

    process_change = pd.DataFrame.from_dict(filename['process_change_over'])
    process = process_change['Time(Mins)'][2]
    for j in range(0,num_machine):
        machine=dict_coating['Coating'][j]
        l=0
        for i in range(0,df.shape[0]):
            if(df.Coating[i]==machine):
               # print(df.Start_Compression[i])
                df['Start_Coating'][i]=datetime.timedelta(minutes=int(process))+df.End_Compression[i]
                df['End_Coating'][i]=df['Start_Coating'][i]+datetime.timedelta(minutes=int(df.Time_Coating[i]))

                if(i>0 and df['End_Coating'][l]!=0.0 and df['Start_Coating'][i]<df['End_Coating'][l]):
                    delay=df['End_Coating'][l]-df['Start_Coating'][i]
                    df['Start_Coating'][i]=df['Start_Coating'][i]+(delay)+datetime.timedelta(minutes=int(process))
                    df['End_Coating'][i]=df['End_Coating'][i]+(delay)+datetime.timedelta(minutes=int(process))
                l=i


    if "time" in filename:
        time = pd.DataFrame.from_dict(filename['time'])
        time['fTime'] = time['date']+time['fTime']
        time['tTime'] = time['date']+time['tTime']

        time['date'] = pd.to_datetime(time['date'])
        
        for j in range(0,time.shape[0]):
            time['fTime'][j] = datetime.datetime.strptime(time['fTime'][j], '%Y-%m-%d%H:%M')
            time['tTime'][j] = datetime.datetime.strptime(time['tTime'][j], '%Y-%m-%d%H:%M')

        for j in range(0,time.shape[0]):
            machine=time['machine'][j]
            for i in range(0,df.shape[0]):
                if(machine==df['Coating'][i]):
                    #print(i,machine)
                    if(df['Start_Coating'][i].date()==time['date'][j].date() or df['End_Coating'][i].date()==time['date'][j].date()):
                        if(df['Start_Coating'][i].date()==time['date'][j].date()):
                            if(df['Start_Coating'][i]>time['fTime'][j] and df['End_Coating'][i]<time['tTime'][j]):
                                print('up',i)
                                diff=time['tTime'][j]-df['Start_Coating'][i]
                                df=update2(machine,diff,i,df)
                        if(df['Start_Coating'][i].date()!=time['date'][j].date() and df['End_Coating'][i].date()==time['date'][j].date()):
                            if(time['fTime'][j]<df['End_Coating'][i]):
                                print(i)
                                #print(df['Start_Granulation'][i],time['tTime'][j])
                                diff=time['tTime'][j]-df['Start_Coating'][i]
                                #print(diff)
                                df=update2(machine,diff,i,df)


    df.sort_values("End_Coating", axis = 0, ascending = True, 
                 inplace = True, na_position ='last') 
    li1=[]
    i=0
    while(i<df.shape[0]):
        for j in range(0,num_of_inspection_machine.shape[0]):
            li1.append(num_of_inspection_machine.Inspection[j])
            i=i+1
    df['Inspection']=li1
    df.sort_values("assigned_priority", axis = 0, ascending = True, 
                 inplace = True, na_position ='last') 
    return inspection(filename,df)


def update3(machine,diff,pos,df):
    for j in range(pos,df.shape[0]):
        if(df['Inspection'][j]==machine):
            #print(df['Start_Compression'][j],df['End_Compression'][j],diff)
            df['Start_Inspection'][j]=diff+df['Start_Inspection'][j]
            #print(df['Start_Compression'][j],df['End_Compression'][j])
            df['End_Inspection'][j]=diff+df['End_Inspection'][j]
            #print(df['Start_Compression'][j],df['End_Compression'][j])
    return df


def inspection(filename,df):
    num_of_inspection_machine=pd.DataFrame.from_dict(filename['inspection_capacity'])
    changeover_time=pd.DataFrame.from_dict(filename['change_over'])
    num_of_granulation_machine=pd.DataFrame.from_dict([{'Granulation': 'G1', 'Rate': 1.5, 'Output_Kgs': 684},
 {'Granulation': 'G2', 'Rate': 1.5, 'Output_Kgs': 684}])
    num_of_granulation_machine.rename(columns={'Output (Kgs)':'Output_Kgs'}, inplace=True)
    dict_granulation=num_of_granulation_machine.to_dict()
    dict_inspection=num_of_inspection_machine.to_dict()
    for i in range(0,changeover_time.shape[0]):
        if(changeover_time.Process[i]=='Inspection'):
            inspection_change=changeover_time['Time (Mins)'][i]
            product_product=changeover_time['Time (Mins)'][i+1]
            break
    num_machine=num_of_inspection_machine.shape[0]
    temp=0
    df['Start_Inspection']=np.zeros(df.shape[0])
    df['Time_Inspection']=np.zeros(df.shape[0])
    df['End_Inspection']=np.zeros(df.shape[0])
    for i in range(0,df.shape[0]):
        for j in range(0,num_machine):
            if(dict_inspection['Inspection'][j]==df.Inspection[i]):
                 break
        df['Time_Inspection'][i]=df.Qty_pcs[i]/dict_inspection['Rate'][j]
    k=0
    for j in range(0,num_machine):
        machine=dict_granulation['Granulation'][j]
        for i in range(0,df.shape[0]):
            if(df.Task[i]==machine):
                #print(i,k)
                if(i-k!=1):
                    df.Start_Inspection[i]=product_product
                    df.End_Inspection[i]=product_product
                k=i
    df.Start_Inspection[0]=df.Start_Inspection[0]-product_product
    df.End_Inspection[0]=df.End_Inspection[1]-product_product

    process_change = pd.DataFrame.from_dict(filename['process_change_over'])
    process = process_change['Time(Mins)'][3]

    df.sort_values("End_Coating", axis = 0, ascending = True, 
                 inplace = True, na_position ='last')
    df.reset_index(inplace=True)

    for j in range(0,num_machine):
        machine=dict_inspection['Inspection'][j]
        l=0
        for i in range(0,df.shape[0]):
            if(df.Inspection[i]==machine):
               # print(df.Start_Compression[i])
                df['Start_Inspection'][i]=datetime.timedelta(minutes=int(process))+df.End_Coating[i]
                df['End_Inspection'][i]=df['Start_Inspection'][i]+datetime.timedelta(minutes=int(df.Time_Inspection[i]))




                if(i>0 and df['End_Inspection'][l]!=0.0 and df['Start_Inspection'][i]<df['End_Inspection'][l]):
                    delay=df['End_Inspection'][l]-df['Start_Inspection'][i]
                    df['Start_Inspection'][i]=df['Start_Inspection'][i]+(delay)+datetime.timedelta(minutes=int(process))
                    df['End_Inspection'][i]=df['End_Inspection'][i]+(delay)+datetime.timedelta(minutes=int(process))
                l=i

    if "time" in filename:
        time = pd.DataFrame.from_dict(filename['time'])
        time['fTime'] = time['date']+time['fTime']
        time['tTime'] = time['date']+time['tTime']

        time['date'] = pd.to_datetime(time['date'])
        
        for j in range(0,time.shape[0]):
            time['fTime'][j] = datetime.datetime.strptime(time['fTime'][j], '%Y-%m-%d%H:%M')
            time['tTime'][j] = datetime.datetime.strptime(time['tTime'][j], '%Y-%m-%d%H:%M')


        for j in range(0,time.shape[0]):
            machine=time['machine'][j]
            for i in range(0,df.shape[0]):
                if(machine==df['Inspection'][i]):
                    #print(i,machine)
                    if(df['Start_Inspection'][i].date()==time['date'][j].date() or df['End_Inspection'][i].date()==time['date'][j].date()):
                        if(df['Start_Inspection'][i].date()==time['date'][j].date()):
                            if(df['Start_Inspection'][i]>time['fTime'][j] and df['End_Inspection'][i]<time['tTime'][j]):
                                print('up',i)
                                diff=time['tTime'][j]-df['Start_Inspection'][i]
                                df=update3(machine,diff,i,df)
                        if(df['Start_Inspection'][i].date()!=time['date'][j].date() and df['End_Inspection'][i].date()==time['date'][j].date()):
                            if(time['fTime'][j]<df['End_Inspection'][i]):
                                print(i)
                                #print(df['Start_Granulation'][i],time['tTime'][j])
                                diff=time['tTime'][j]-df['Start_Inspection'][i]
                                #print(diff)
                                df=update3(machine,diff,i,df)


    df = df[['id', 'Task', 'Products', 'Batches', 'Qty_pcs',
       'assigned_priority', 'shift', 'Qty_Kgs', 'Start_Granulation',
       'Time_Granulation', 'End_Granulation', 'Compression',
       'Start_Compression', 'Time_Compression', 'End_Compression', 'Coating',
       'Start_Coating', 'Time_Coating', 'End_Coating', 'Inspection',
       'Start_Inspection', 'Time_Inspection', 'End_Inspection']]
   

    df.rename(columns={'Task':'Task_Granulation','Qty_Kgs':'Qty_kgs','Time_Granulation':'Time','Start_Granulation':'start','End_Granulation':'end', 'Compression':'Task_Compression','Coating' : 'Task_Coating','Inspection':'task_inspection', 'Start_Compression':'start_compression','Time_Compression':'time_compression', 'End_Compression':'end_compression', 'Start_Coating':'start_coating', 'Time_Coating':'time_coating', 'End_Coating':'end_coating','Start_Inspection':'start_inspection', 'Time_Inspection':'time_inspection', 'End_Inspection':'end_inspection'},inplace=True)


    

       


    granulation_resource = ['Granulation I', 'Granulation II']
    for i in range(len(granulation_resource)):
        df = df.replace(to_replace='G'+str(i+1), value=granulation_resource[i])


    delay_demand = pd.DataFrame.from_dict(filename['demand'])

    df['batch_delayed'] = 0
    for index, row in df.iterrows():
        df['batch_delayed'][index] = (df['end_inspection'][index] - pd.to_datetime(delay_demand.loc[delay_demand['Products ']== df['Products'][index]]['Due date'])).dt.days

    df['batch_delayed'] = df['batch_delayed'].clip(lower=0) #for removing negatives batch delay

    result_dict = {}

    result_dict['schedule'] = df.to_dict('record')

    demand_shortage = pd.DataFrame.from_dict(filename['demand_unconstrained'])[['id', 'Products', 'Order Qty(pcs)', 'Qty (kgs)', 'Due date', 'Remaining_Qty(in Kgs)']]
    demand_shortage.rename(columns={'Remaining_Qty(in Kgs)':'Shortage(in Kgs)'}, inplace=True)
    demand_shortage['Shortage(in Kgs)'] = demand_shortage['Shortage(in Kgs)'].clip(lower=0)

    result_dict['shortage'] = demand_shortage.to_dict('record')

    return result_dict

#final_push

@api_view(['POST'])
def model_view(request):
    data_dict = request.data
    # print(data_dict)

    json_data, task_df = schedule_granulation(data_dict)
    final_data = input(json_data, task_df)

    return Response(final_data)