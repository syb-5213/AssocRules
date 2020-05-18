import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import  linear_model
import itertools
import math

print("读取数据中...")
data = pd.read_csv('cbg_patterns.csv')
print("读取完毕。")
print("属性有: census_block_group、date_range_start、date_range_end、raw_visit_count、raw_visitor_count、visitor_home_cbgs、visitor_work_cbgs、distance_from_home、related_same_day_brand、related_same_month_brand、top_brands、popularity_by_hour、popularity_by_day")

#提取数据集中数值属性
v_title=["raw_visit_count","raw_visitor_count","distance_from_home"]
val_data=data[v_title]
print("数值属性有:raw_visit_count、raw_visitor_count、distance_from_home;")
#提取数据集中标称属性
c_title=["related_same_day_brand"]
print("标称属性有:census_block_group、date_range_start、date_range_end、visitor_home_cbgs、visitor_work_cbgs、related_same_day_brand、related_same_month_brand、top_brands、popularity_by_hour、popularity_by_day;")
 
#列出标称属性related_same_day_brand的频度图
print("标称属性related_same_day_brand频度图(隐去横坐标，并且由于各种类品牌数据并不是直接给出,所以先转换格式后再统计个数):")						
a=data["related_same_day_brand"]
all_data=[]
for i in range(len(a)):
    b=a.iloc[[i]].values[0]
    c=b[1:len(b)-1]
    if(not(len(c)==0)):
        t_data=[]
        d=c.split(',')
        for j in range(len(d)):
            e=d[j][1:len(d[j])-1]
            t_data.append(e)
        all_data.append(t_data)
#首先将数据集变为易于挖掘关联规则的格式，如下列出一些转换后的数据格式，每一项记录代表一个人访问过哪些品牌
for i in range(len(all_data)):
    if(i<10):
        print(all_data[i])
    else:
        break
"""    
item_value ={} 
for i in range(len(all_data)):
    for j in range(len(all_data[i])):
        if all_data[i][j] in item_value:
            item_value[all_data[i][j]]=item_value[all_data[i][j]]+1
        else:
            item_value[all_data[i][j]]=1
new_item_value=sorted(item_value.items(),key=lambda x:x[1],reverse=True)
for i in range(len(all_data)):
    all_data[i]=sorted(all_data[i], key=lambda x:item_value[x],reverse=True)

print("列出访问人数最多的前20中品牌及其访问次数，可以发现访问次数最多的是mcdonalds，总共有46457次。")
print(new_item_value[0:20])

"""

#创建node类作为fp树的每个结点，其中name用于存放结点名字，count用于计数，nodeLink用于连接相似结点，parent用于存放父节点，用于回溯，children存放儿子结点。
class node:
    def __init__(self, name, counts, parent_node):
        self.name = name
        self.count = counts
        self.nodeLink = None
        self.parent = parent_node
        self.children = {}

    def add(self, counts):
        self.count += counts

    def disp(self, ind=1):
        print('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)
            
def update_item_list(node1, node2):
    while node1.nodeLink != None:
        node1 = node1.nodeLink
    node1.nodeLink = node2
def update_FPtree(items, tree, item_list, count):
    if items[0] in tree.children:
        #判断items的第一个结点是否已作为子结点
        tree.children[items[0]].add(count)
    else:
        #创建新的分支
        tree.children[items[0]] = node(items[0], count, tree)
        #更新相应频繁项集的链表，往后添加
        if item_list[items[0]][1] == None:
            item_list[items[0]][1] = tree.children[items[0]]
        else:
            update_item_list(item_list[items[0]][1], tree.children[items[0]])
    #递归
    if len(items) > 1:
        update_FPtree(items[1::], tree.children[items[0]], item_list, count)

def crerat_fp_tree(data, minSupport):
    #创建每个频繁item的列表item_list
    item_list = {}
    #根据数据集data汇集每个item数量
    for i in data:
        for j in i:
            item_list[j] = item_list.get(j, 0)+data[i]
    #将数据项小于min_support删除掉
    for k in list(item_list.keys()):
        if item_list[k] < minSupport:
            del(item_list[k])
    #创建频繁项集的索引
    frequent_item = set(item_list.keys())
    #将item_list的格式改为值及指向下一个item出现处的指针
    for k in item_list:
        item_list[k] = [item_list[k], None]
    #创建FP树的树根tree结点
    tree = node('', 1, None)
    for i,count in data.items():
        #首先筛选出满足最小支持度的项集
        item = {}
        for j in i:
            if j in frequent_item: 
                item[j] = item_list[j][0] # element : count
        if len(item) > 0:
            #按照频数对每个item项集进行排序并更新fp树
            ordered_item = [v[0] for v in sorted(item.items(), key=lambda p:(p[1], -ord(p[0][0])), reverse=True)]
            update_FPtree(ordered_item, tree, item_list, count)
    return tree, item_list

#合并相同项集并用count记录该项集出现的次数
def create_comb_data(data):
    comb_data={}
    for i in data:
        key = frozenset(i)
        if key in comb_data:
            comb_data[frozenset(i)] += 1
        else:
            comb_data[frozenset(i)] = 1
    return comb_data


def loadSimpDat():
    simDat = [['f','a','c','d','g','i','m','p'],
              ['a','b','c','f','l','m','o'],
              ['b','f','h','j','o','w'],
              ['b','c','k','s','p'],
              ['a','f','c','e','l','p','m','n']]
    return simDat
minSupport=100
minConfidence=0.75
simdat=loadSimpDat()
comb_data = create_comb_data(all_data)
#根据每个项集出现的频数，设置最小支持度为minSupport进行fp树的构建
tree, item_list = crerat_fp_tree(comb_data, minSupport)


# 构造指定item的条件模式基
def create_conditional_database(item, item_list):
    #tree_node变量指向传参item在FP树中的第一个结点
    tree_node = item_list[item][1] 
    conditional_database={}
    #递归的沿着tree_node结点向上寻找一条到根节点的路径并保存在path中，之后在进行下一个item位置继续寻找path
    while tree_node != None:
        path = []
        temp_node=tree_node
        if(temp_node.parent!=None):
            while temp_node.parent!=None:
                path.append(temp_node.name)
                temp_node=temp_node.parent
            if len(path) > 1:
                conditional_database[frozenset(path[1:])] = tree_node.count # 关联treeNode的计数
            tree_node = tree_node.nodeLink # 下一个basePat结点
    return conditional_database



#对每一个频繁项都建立一棵条件FP树，然后对每个条件FP树递归地挖掘，最后可以得到所有满足最小支持度的频繁项集。
def mine_FPtree(tree, item_list, minSupport, l, result):
    #最开始的频繁项集是item_list中的各元素
    t = [v[0] for v in sorted(item_list.items(), key=lambda p:p[1][0])]
    #对每个频繁项进行处理
    for i in t: 
        new_t = l.copy()
        new_t.append(i)
        if(len(new_t)>1):
            result.append([new_t,item_list[i][0]])
        #当前频繁项集的条件模式基
        conditional_database_i = create_conditional_database(i, item_list) 
        #构造当前频繁项的条件FP树
        new_tree, new_item_list = crerat_fp_tree(conditional_database_i, minSupport) 
        if new_item_list != None:
            #递归挖掘条件FP树
            mine_FPtree(new_tree, new_item_list, minSupport, new_t, result) 
            
freqItems = []
mine_FPtree(tree, item_list, minSupport, list([]), freqItems)
#筛选出其中30个频繁项集以及其支持度进行展示，
for i in range(len(freqItems)):
    if(i<30):
        print(freqItems[i][0],freqItems[i][1])
    else:
        break
assocrules = []
for i in range(len(freqItems)):
    t=freqItems[i][0]
    count=freqItems[i][1]
    for j in range(len(t)):
        item=t[j]
        count_item=item_list[item][0]
        if(count/count_item>minConfidence):
            assocrules.append([item,t,count,count/count_item])
#设置参数最小支持度为00，最小置信度为0.75提取关联规则，最终得到61组关联规则,其中每一条规则有4项，第一个为条件，第二项为全集，第三项为支持度，第四项为置信度
#例如["Buddy's", ["Buddy's", 'Stripes Convenience Stores'], 119, 0.9296875]表示Buddy's->Stripes Convenience Stores[119,0.9296875]
for i in range(len(assocrules)):
    print(assocrules[i])            

#采用Lift以及Allconf两种指标对提取出来的关联规则进行评价
sum_number=len(all_data)
for i in range(len(assocrules)):
    sab=assocrules[i][2]/sum_number
    sa=item_list[assocrules[i][1][0]][0]/sum_number
    sb=item_list[assocrules[i][1][1]][0]/sum_number
    lift=sab/(sa*sb)
    allconf=(sab/sa+sab/sb)/2
    assocrules[i]=[assocrules[i][0],assocrules[i][1],assocrules[i][2],assocrules[i][3],lift,allconf]
  
#筛选出其中一些关联规则进行分析
temp=sorted(assocrules, key=lambda p:p[2], reverse=True)  
#根据该条关联规则显示，支持度为622，置信度为77.9%，即既访问Cracker Barrel同时又访问mcdonalds的人数共有622人次，在访问Cracker Barrel的人群中有77.9%的人同样会访问mcdonalds，
#该条规则的评价lift指标为3大于1表示具有正相关关系，而allconf指标显示为0.01小于0.5应该为负相关，这说明该规则受到零事务的影响导致lift评价指标出现问题，由该项规则可以发现lift指标受零事务的影响很大，而实际上我们去Cracker Barrel和去McDonald的相关性不应该取决于两者都不访问的记录，这一点正如课上所讲的lift和卡方指标的劣势，即容易受到数据记录大小的影响。
print(temp[11]) 
#根据该条关联规则显示，支持度为180，置信度为94.2%，既访问Jeep同时又访问Dodge的人数共有188人次，在访问Jeep的人群中有94.2%的人同样会访问Dodge，
#该条规则的评价lift指标为863远远大于1表示具有正相关关系，而allconf指标显示为0.89大于0.5同样表示正相关关系，这说明jeep和dodge确实存在着强烈的正相关关系，而在实际中Jeep和Dodge是两款车的品牌都是克莱斯勒的产品，基本是同一个平台，所以两个品牌之间有非常强的共性，想购买Jeep车的人往往也会去Dodge看看，这一点比较贴合实际生活。所以可以在浏览Jeep的人群中宣传Dodge可能是一件比较有效率销售策略。
print(temp[36]) 

