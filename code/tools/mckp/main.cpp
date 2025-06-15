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

Dataset readFile(string fileName, double weights[], int size_weights);
void writeFile(string fileName_out, string fileName_id, Allocation optimalAllocation);
void writeFile(string fileName_out, string fileName_id, WeightedAllocation optimalAllocation);

Dataset readFile(string fileName, double weights[], int size_weights){
	ifstream fin(fileName);

	vector<double> weightClass;
    weightClass.assign(weights, weights+size_weights);

	vector< vector< vector<double> > > datasetValues;
	string pred_prob_all = "";

	getline(fin, pred_prob_all); // 表头
	while(getline(fin, pred_prob_all)){ // 数据
		vector<double> valueClass;
		vector< vector<double> > eachClass;
		double pred_prob = 0.0;

		stringstream ss(pred_prob_all);
	    while(ss>>pred_prob){
	    	valueClass.push_back(pred_prob);
	    }

    	eachClass.push_back(valueClass);
		eachClass.push_back(weightClass);
		datasetValues.push_back(eachClass);
	}
	
	fin.close();

	Dataset data(datasetValues);
	return data;
}

void writeFile(string fileName_out, string fileName_id, Allocation optimalAllocation){
	ifstream fin(fileName_id);
	ofstream fout(fileName_out, ios::trunc);

	fout << "id" << "\t" << "index" << "\t" << "treatment" << "\t" << "lift" << endl;
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

void writeFile(string fileName_out, string fileName_id, WeightedAllocation optimalAllocation){
	// ifstream fin(fileName_id);
	ofstream fout(fileName_out, ios::trunc);

	fout << "index" << "\t" << "treatment" << "\t" << "lift" << "\t" << "proportion" << endl;
	vector<Item*> items = optimalAllocation.getItems();
	vector<double> proportions = optimalAllocation.getProportions();
	// string id = "";

	// getline(fin, id); // 表头
	for(unsigned int i=0;i<items.size();i++){
		if(proportions[i]==1){
            fout << items[i]->getIndex() << "\t" << items[i]->getWeight() << "\t" << items[i]->getValue() << endl;
        }
        else{
        	if(i!=items.size()){
        		fout << items[i]->getIndex() << "\t" << items[i]->getWeight() << "\t" << items[i]->getValue() << "\t" << proportions[i] << endl;
                fout << items[i]->getIndex() << "\t" << items.back()->getWeight() << "\t" << items.back()->getValue() << "\t" << proportions.back() << endl;
            }
        }
	}

	// fin.close();
	fout.close();
}

int main(){
	string fileName_id = "df_sample_pred_unique_id.txt";
	string fileName_y = "df_sample_pred_unique_y.txt";
	string fileName_out = "data/df_sample_res.txt";
	double weights[] = {1.0, 2.0, 3.0, 4.0, 5.0};
	double cac = 3.0;
	int mode = 3;
	map<int, string> map_mode = {
		{1, "Exhaustive Search"}, 
		{2, "Discrete"}, 
		{3, "Dyer-Zemel"}, 
		{4, "Greedy"}
	};
	
	cout << "dataSet read..." << endl;
	Dataset ds = readFile(fileName_y, weights, sizeof(weights)/sizeof(weights[0]));
	cout << "dataSet read done!!!" << endl;

	int ds_size = ds.getNbClasses();
	cout << "dataSet size: " << ds_size << endl;
	cout << "CAC: " << cac << endl;
	double capacity = ds_size * cac;
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
