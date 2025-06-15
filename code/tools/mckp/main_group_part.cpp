// -*- coding: utf-8 -*-
#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <string>

#include "data.h"
#include "ExhaustiveSearch.h"
#include "GreedyAlgorithm.h"
#include "DyerZemel.h"
#include "DiscreteAlgorithm.h"

#define SMALL
#define LARGE

using namespace std;

Dataset readFile(string fileName_y, string fileName_cnt, string fileName_weight);
void writeFile(string fileName_out, string fileName_id, Allocation optimalAllocation);

Dataset readFile(string fileName_y, string fileName_cnt, string fileName_weight){
    ifstream fin_y(fileName_y);
    ifstream fin_cnt(fileName_cnt);
    ifstream fin_wgt(fileName_weight);
    
    vector< vector< vector<double> > > datasetValues;
    string pred_prob_all = "";
    string group_cnt = "";
    string group_weight = "";
    
    getline(fin_cnt, group_cnt); // 表头
    while(getline(fin_y, pred_prob_all)&&getline(fin_cnt, group_cnt)&&getline(fin_wgt, group_weight)){ // 数据
        vector<double> valueClass;
        vector<double> weightClass;
        vector< vector<double> > eachClass;
        double pred_prob = 0.0;
        double cnt = 0.0;
        double wgt = 0.0;
        
        stringstream ss_y(pred_prob_all);
        while(ss_y>>pred_prob){
            valueClass.push_back(pred_prob);
        }
        eachClass.push_back(valueClass);
        
        stringstream ss_cnt(group_cnt);
        ss_cnt >> cnt;
        
        stringstream ss_wgt(group_weight);
        while(ss_wgt>>wgt){
            weightClass.push_back(wgt*cnt);
        }
        eachClass.push_back(weightClass);
        
        datasetValues.push_back(eachClass);
    }
    
    fin_y.close();
    fin_cnt.close();
    fin_wgt.close();
    
    Dataset data(datasetValues);
    return data;
}

void writeFile(string fileName_out, string fileName_id, Allocation optimalAllocation){
    ifstream fin(fileName_id);
    ofstream fout(fileName_out, ios::trunc);
    
    fout << "id" << "\t" << "index" << "\t" << "treatment" << "\t" << "uplift" << endl;
    vector<Item*> items = optimalAllocation.getItems();
    string id = "";
    
    getline(fin, id); // 表头
    for(auto e: items){
        getline(fin, id); // 数据
        fout << id << "\t" << e->getIndex() << "\t" << e->getWeight() << "\t" << e->getValue() << endl;
    }
    
    fin.close();
    fout.close();
}

int main(){
    string fileName_id = "df_csg_c_sum_correct_id.txt";
    string fileName_y = "df_csg_s_sum_correct_limit_part.txt";
    string fileName_cnt = "df_csg_cnt_sum_correct.txt";
    string fileName_weight = "df_csg_weight_sum_correct_limit_part.txt";
    string fileName_out = "df_csg_result_sum_correct_limit_part.txt";
    
    double cac = 1.0;
    int people_size = 61759069;
    int mode = 3;
    map<int, string> map_mode = {
        {1, "Exhaustive Search"}, 
        {2, "Discrete"}, 
        {3, "Dyer-Zemel"}, 
        {4, "Greedy"}
    };
    
    cout << "dataSet read..." << endl;
    Dataset ds = readFile(fileName_y, fileName_cnt, fileName_weight);
    cout << "dataSet read done!!!" << endl;
    
    cout << "data size(real): " << people_size << endl;
    cout << "CAC: " << cac << endl;
    double capacity = people_size * cac;
    cout << "capacity: " << capacity << endl;
    
    cout << "LP begin..." << endl;
    if(mode==1){
        cout << "===== " << map_mode[mode] << " =====" << endl;
        Allocation optimalAllocation = ExhaustiveSearch(&ds, capacity);
        cout << "Optimal value is: " << optimalAllocation.getValue() << endl;
        cout << "Final weight is: " << optimalAllocation.getWeight() << endl;
        cout << "write file..." << endl;
        writeFile(fileName_out, fileName_id, optimalAllocation);
        cout << "write file done!!!" << endl;
    } 
    else if(mode==2){
        cout << "===== " << map_mode[mode] << " =====" << endl;
        Allocation optimalAllocation = MCKP_Discrete_Algorithm(&ds, capacity);
        cout << "Optimal value is: " << optimalAllocation.getValue() << endl;
        cout << "Final weight is: " << optimalAllocation.getWeight() << endl;
        cout << "write file..." << endl;
        writeFile(fileName_out, fileName_id, optimalAllocation);
        cout << "write file done!!!" << endl;
    }
    else if(mode==3){
        cout << "===== " << map_mode[mode] << " =====" << endl;
        pair<double, Allocation> resultPair = DyerZemelAlgorithm(&ds, capacity);
        Allocation optimalAllocation = resultPair.second;
        cout << "Optimal value is: " << optimalAllocation.getValue() << endl;
        cout << "Final weight is: " << optimalAllocation.getWeight() << endl;
        cout << "write file..." << endl;
        writeFile(fileName_out, fileName_id, optimalAllocation);
        cout << "write file done!!!" << endl;
    }
    else if(mode==4){
        cout << "===== " << map_mode[mode] << " =====" << endl;
        WeightedAllocation optimalAllocation = MCKP_Greedy_Algorithm(&ds, capacity);
        cout << "Optimal value is: " << optimalAllocation.getValue() << endl;
        cout << "Final weight is: " << optimalAllocation.getWeight() << endl;
        cout << "write file..." << endl;
        writeFile(fileName_out, fileName_id, optimalAllocation);
        cout << "write file done!!!" << endl;
    }
    cout << "LP done!!!" << endl;
    
    return 0;
}
