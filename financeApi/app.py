import flask, json
import datetime
from flask import request
from jqdatasdk import *
from jqdatasdk.api import get_money_flow, get_price,get_fundamentals
from jqdatasdk import finance
import pandas as pd
import numpy as np
from pandas.io.parquet import FastParquetImpl

auth('13739188902', 'ZNnb20160801')
recentNoun=['最近','最近几天','最近十天','recent','近期','近几天','这几天']
def indexAnalyse(asset,date,code):
    today=datetime.date.today()
    priceD=get_price(code,start_date=(today-datetime.timedelta(days=100)).strftime('%Y-%m-%d'),end_date=today.strftime('%Y-%m-%d'))

    messages=''
    if date=='today' or date =='今天':
        tradeDate=datetime.date.today()
        date ='今天'
    elif date=='yesterday' or date =='昨天':
        tradeDate=datetime.date.today()-datetime.timedelta(days=1)
        date ='昨天'
    elif date in recentNoun:
        tradeDate=priceD.index.tolist()[-1]
        date='recent'
    elif date=='最近一个交易日':
        tradeDate=priceD.index[-1]
    else:
         messages+='对不起，我正在学习当中。'
         resu={'code':111,'data':messages}
         return resu
 
    if tradeDate not in priceD.index.tolist():
        messages='该日不是交易日或数据还未更新，为你播报最近日期{}数据;'.format(priceD.index[-1].strftime('%Y-%m-%d'))
        tradeDate=priceD.index[-1]
        
    openP=round(priceD.loc[tradeDate,'open'],2)
    closeP=round(priceD.loc[tradeDate,'close'],2)
    highP=round(priceD.loc[tradeDate,'high'],2)
    lowP=priceD.loc[tradeDate,'low']
    vol=priceD.loc[tradeDate,'money']
    avgP=np.average(priceD['close'].tolist()[-10:-1])
    profit=priceD['close'].rolling(2).apply(lambda x:(x[1]-x[0])/x[0]).fillna(0)
    if date=='recent':
        pl=profit.tolist()[-10:len(profit)]
        posDay=sum([1 if i>0 else 0 for i in pl])
        messages+='近十天'+asset
        if posDay>=5:
            messages+='以涨居多，涨幅最大的一天达'+str(round(100*max(pl),1))+'%。'
        else:
            messages+='以跌居多，涨幅最大的一天达'+str(round(100*max(pl),1))+'%。'
        messages+='最近一个交易日，'+asset
        if pl[-1]>0:
            messages+='上涨'+str(round(100*pl[-1],1))+'%。'
        else:
            messages+='下跌'+str(round(-100*pl[-1],1))+'%。'
        print(messages)
        resu={'code':111,'data':messages}
        return resu
    messages+=asset+date+'收盘价为{}，'.format(closeP)
    change=profit[tradeDate]
    if change>0:
        messages+='上涨'+str(round(100*change,1))+'%。'
    else:
        messages+='下跌'+str(round(-100*change,1))+'%。'
    messages+='最高价位{}，最低价为{}，成交金额为{}万亿， 10日均线价格为{}'.format(highP,round(lowP,2),round(vol/10e9,2),round(avgP,2),'.2f')
    print(messages)
    resu={'code':111,'data':messages}
    return resu

def stockAnalyse(asset,date,code):
    today=datetime.date.today()
    messages=''
    priceD=get_price(code,start_date=(today-datetime.timedelta(days=100)).strftime('%Y-%m-%d'),end_date=today.strftime('%Y-%m-%d'))
    moneyD=get_money_flow([code],start_date=(today-datetime.timedelta(days=100)).strftime('%Y-%m-%d'),end_date=today.strftime('%Y-%m-%d'),\
        fields=['date','change_pct','net_amount_main','net_pct_main'])
    if date=='today' or date =='今天':
        tradeDate=datetime.date.today()
        date ='今天'
    elif date=='yesterday' or date =='昨天':
        tradeDate=datetime.date.today()-datetime.timedelta(days=1)
        date ='昨天'
    elif date in recentNoun:
        tradeDate=moneyD.iloc[-1,0]
        date='recent'
    elif date=='最近一个交易日':
        tradeDate=moneyD.iloc[-1,0]
    else:
         messages+='对不起，我正在学习当中。'
         resu={'code':111,'data':messages}
         return resu
    
    if tradeDate not in priceD.index.tolist() or tradeDate not in moneyD['date'].tolist() :
        messages='该日不是交易日或数据还未更新完毕，为你播报最近日期{}数据;'.format(priceD.index[-1].strftime('%Y-%m-%d'))
        tradeDate=priceD.index[-1]
    moneyD.set_index('date',inplace=True)
    openP=round(priceD.loc[tradeDate,'open'],2)
    closeP=round(priceD.loc[tradeDate,'close'],2)
    highP=round(priceD.loc[tradeDate,'high'],2)
    lowP=priceD.loc[tradeDate,'low']
    vol=priceD.loc[tradeDate,'money']
    net_amount_main=moneyD.loc[tradeDate,'net_amount_main']
    net_pct_main=moneyD.loc[tradeDate,'net_pct_main']
    avgP=np.average(priceD['close'].tolist()[-10:-1])
    profit=priceD['close'].rolling(2).apply(lambda x:(x[1]-x[0])/x[0]).fillna(0)
    if date=='recent':
        pl=moneyD['change_pct'].tolist()[-10:len(profit)]
        main_power=moneyD['net_amount_main'].tolist()[-10:len(profit)]
        posDay=sum([1 if i>0 else 0 for i in pl])
        messages+='近十天'+asset
        if posDay>=5:
            messages+='以涨居多，涨幅最大的一天达'+str(round(max(pl),1))+'%。'
        else:
            messages+='以跌居多，涨幅最大的一天达'+str(round(max(pl),1))+'%。'
        messages+='最近一个交易日，'+asset
        if pl[-1]>0:
            messages+='上涨'+str(round(pl[-1],1))+'%。'
        else:
            messages+='下跌'+str(round(-pl[-1],1))+'%。'
        net_amount_main=sum(main_power)
        if net_amount_main>0:
            messages+='近十个交易日主力净流入'+str(round(net_amount_main,2))+'万'
        else:
            messages+='近十个交易日主力净流出'+str(-round(net_amount_main,2))+'万'
        print(messages)

        resu={'code':111,'data':messages}
        return resu
    messages+=asset+date+'收盘价为{}，'.format(closeP)
    change=profit[tradeDate]
    if change>0:
        messages+='上涨'+str(round(100*change,1))+'%。'
    else:
        messages+='下跌'+str(round(-100*change,1))+'%。'
    messages+='最高价位{}，最低价为{}，成交金额为{}亿， 10日均线价格为{}。'.format(highP,round(lowP,2),round(vol/10e5,2),round(avgP,2))
    if net_amount_main>0:
        messages+='主力净流入'+str(round(net_amount_main,2))+'万'
        messages+=',主力净占比'+str(round(net_pct_main,2))+'%。'
    else:
        messages+='主力净流出'+str(-round(net_amount_main,2))+'万'
        messages+=',主力净占比'+str(round(net_pct_main,2))+'%。'
    print(messages)
    resu={'code':111,'data':messages}
    return resu

def fundAnalyse(asset,date,code):
    messages=''
    priceD=finance.run_query(query(finance.FUND_NET_VALUE).filter(finance.FUND_NET_VALUE.code==code).order_by(finance.FUND_NET_VALUE.day.desc()).limit(250))
    priceD.set_index('day',inplace=True)
    priceD.sort_index(inplace=True,ascending=True)
    if date=='today' or date =='今天':
        tradeDate=datetime.date.today()
        date ='今天'
    elif date=='yesterday' or date =='昨天':
        tradeDate=datetime.date.today()-datetime.timedelta(days=1)
        date ='昨天'
    elif date=='最近一个交易日':
        tradeDate=priceD.index[-1]
    elif date in recentNoun:
        date='recent'
        tradeDate=priceD.index[-1]
    else:
         messages+='对不起，我正在学习当中。'
         resu={'code':111,'data':messages}
         return resu
    start_date=tradeDate.strftime('%Y-%m-%d').split('-')
    start_date[0]=str(int(start_date[0])-1)
    start_date=start_date[0]+'-'+start_date[1]+'-'+start_date[2]
 
    if tradeDate not in priceD.index.tolist():
        messages='该日不是交易日或数据还未更新，为你播报最近日期{}数据;'.format(priceD.index[-1].strftime('%Y-%m-%d'))
        tradeDate=priceD.index[-1]
        date='该日'
    sum_nv=round(priceD.loc[tradeDate,'sum_value'],2)
    nv=round(priceD.loc[tradeDate,'net_value'],2)
    profit=priceD['net_value'].rolling(2).apply(lambda x:(x[1]-x[0])/x[0]).fillna(0)
    yearprofit=(priceD['net_value'].tolist()[-1]-priceD['net_value'].tolist()[0])/priceD['net_value'].tolist()[0]
    yearmaxDD=MaxDrawdown(priceD['net_value'].tolist())
    yearsharpe=getSharpe(profit)
    if date=='recent':
        pl=profit.tolist()[-10:len(profit)]
        posDay=sum([1 if i>0 else 0 for i in pl])
        messages+='近十天'+asset
        if posDay>=5:
            messages+='以涨居多，涨幅最大的一天达'+str(round(100*max(pl),1))+'%。'
        else:
            messages+='以跌居多，跌幅最大的一天达'+str(round(100*max(pl),1))+'%。'
        messages+='最近一个交易日，'+asset
        if pl[-1]>0:
            messages+='上涨'+str(round(100*pl[-1],1))+'%。'
        else:
            messages+='下跌'+str(round(-100*pl[-1],1))+'%。'
        print(messages)
        resu={'code':111,'data':messages}
        return resu
    messages+=asset+date+'单位净值为{}，'.format(nv)
    change=profit[tradeDate]

    if change>0:
        messages+='上涨'+str(round(100*change,1))+'%。'
    else:
        messages+='下跌'+str(round(-100*change,1))+'%。'
    messages+='基金近一年最大回撤为{}%,收益率为{}%,夏普比率为{}.'.format(round(yearmaxDD*100,2),round(yearprofit*100,2),round(yearsharpe,2))
    print(messages)
    resu={'code':111,'data':messages}
    return resu

def stockDetail(asset,date_range,info,code):
    today=datetime.date.today()
    priceD=get_price(code,start_date=(today-datetime.timedelta(days=300)).strftime('%Y-%m-%d'),end_date=today.strftime('%Y-%m-%d'))
    messages=''
    q = query(
    valuation
    ).filter(
    valuation.code == code
    )
    df = get_fundamentals(q)
    print(df['market_cap'][0])
    base_info=finance.run_query(query(finance.STK_COMPANY_INFO).filter(finance.STK_COMPANY_INFO.code==code).limit(1))
    if date_range=='最近一周'or date_range=='上周':
        priceD=priceD.iloc[-5:len(priceD.index),:]
    elif date_range=='最近一月' or date_range=='最近一个月'or date_range=='上月':
        priceD=priceD.iloc[-22:len(priceD.index),:]
    elif date_range=='最近一年'  or date_range=='今年':
        priceD=priceD.iloc[-250:len(priceD.index),:]
    elif date_range=='NULL':
        pass
    else: 
        messages+='不好意思，我正在学习中'
        return {'code':111,'data':messages}
    if info=='收益率' or info=='涨跌':
        profit=(priceD['close'].tolist()[-1]-priceD['close'].tolist()[0])/(priceD['close'].tolist()[0])
        messages+=asset+date_range+'的收益率为{}%。'.format(round(profit*100,2))
    elif info=='换手率':
        totalCapital=df['market_cap'][0]
        turnoverrate=df['turnover_ratio'][0]
        messages+=asset+'的换手率为{}。'.format(round(turnoverrate,2))
    elif info=='财务指标':
        messages+='最新数据显示，'+asset
        pb=df['pb_ratio'][0]
        pe=df['pe_ratio'][0]
        mv=df['market_cap'][0]
        messages+='总市值为{}亿元，市盈率为{}，市净率为{}'.format(round(mv,2),round(pe,2),round(pb,2))
    elif info=='行业':
        industry=base_info['industry_2'][0]
        business=base_info['main_business'][0]
        comments=base_info['comments'][0]
        if comments is None:
            comments='是A股上市公司。'
        messages+=asset+'所属二级行业为{}，主营业务为{}；{}'.format(industry,business,comments)
    else:
        messages+='不好意思，我正在学习中'
    resu={'code':111,'data':messages}
    return resu

def fundDetail(asset,date_range,info,code):
    messages=''
    priceD=finance.run_query(query(finance.FUND_NET_VALUE).filter(finance.FUND_NET_VALUE.code==code).order_by(finance.FUND_NET_VALUE.day.desc()))
    priceD.set_index('day',inplace=True)
    priceD.sort_index(inplace=True,ascending=True)
    if date_range=='最近一周' or date_range=='上周':
        priceD=priceD.iloc[-5:len(priceD.index),:]
    elif date_range=='最近一月' or date_range=='上月' or date_range=='最近一个月':
        priceD=priceD.iloc[-22:len(priceD.index),:]
    elif date_range=='最近一年' or date_range=='今年':
        priceD=priceD.iloc[-250:len(priceD.index),:]
    elif date_range=='成立以来':
        pass
    elif date_range=='NULL':
        pass
    else: 
        messages+='不好意思，我正在学习中'
        return {'code':111,'data':messages}
    profit=priceD['net_value'].rolling(2).apply(lambda x:(x[1]-x[0])/x[0]).fillna(0)
    yearprofit=(priceD['net_value'].tolist()[-1]-priceD['net_value'].tolist()[0])/priceD['net_value'].tolist()[0]
    maxDD=MaxDrawdown(priceD['net_value'].tolist())
    yearsharpe=getSharpe(profit)
    if info=='收益率' or info=='超额收益':
        messages+=asset+date_range+'的收益率为{}%。'.format(round(yearprofit*100),2)
    elif info=='最大回撤':
        messages+=asset+date_range+'的最大回撤为{}%。'.format(round(maxDD*100),2)
    elif info=='夏普比率':
        messages+=asset+date_range+'的夏普比率为{}。'.format(round(yearsharpe),2)
    else:
        messages+='不好意思，我正在学习中'
    return {'code':111,'data':messages}

def MaxDrawdown(return_list):
    '''最大回撤率'''
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 结束位置
    if i == 0:
        return 0
    j = np.argmax(return_list[:i])  # 开始位置
    return (return_list[j] - return_list[i]) / (return_list[j])

def getSharpe(profit:pd.Series):
    
    return profit.mean()/profit.std()*np.sqrt(252)