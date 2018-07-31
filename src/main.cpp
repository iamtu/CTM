#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <random>

#include <time.h>
#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>

#include "toml.h"
#include "parallel.h"

using namespace std;

using std::string;
using std::vector;


std::vector<string> string_split(const string& s, const string &c) {
	std::vector<string> v;
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while(string::npos != pos2){
    	v.push_back(s.substr(pos1, pos2-pos1));
      	pos1 = pos2 + c.size();
      	pos2 = s.find(c, pos1);
    }
    if(pos1 != s.length())
      	v.push_back(s.substr(pos1));
    return v;
}

std::string string_join(std::vector<string>& elements, std::string delimiter) {
    std::stringstream ss;
    size_t elems = elements.size(),
    last = elems - 1;

    for( size_t i = 0; i < elems; ++i ) {
        ss << elements[i];
        if( i != last )
          ss << delimiter;
    }
    return ss.str();
}



struct Document {
	string doc_id;
    int* words;
    double* counts;
    double total;
	int length;
	double obj, likelihood;
	double *a;

public:
	Document(int length) {
		this->length = length;
		this->words = new int[length];
		this->counts = new double[length];
	}
	~Document(){
		delete[] words;
		delete[] counts;
	}
};


struct Corpus {
	std::vector<Document*> docs;
    int num_terms;
public:
	Corpus(){};

	void read_data(std::string location){
        cout << "REMOVEME - reading data " << endl;
		FILE* file = fopen(location.c_str(),"r");
    	char* line = NULL;
    	size_t len = 0;
    	ssize_t read;

		int max_word_id = -1;

    	while((read=getline(&line,&len,file)) != -1) {
			std::vector<string> s = string_split(std::string(line), " ");
			int length = (int) (s.size() - 1);
			Document* doc = new Document(length);
			doc->doc_id = s[0];
			int total = 0;
			for(size_t i = 1; i < s.size(); i++){
				std::vector<string> s1 = string_split(s[i], ":");
				int word_id = std::stoi(s1[0]);
				int count = std::stoi(s1[1]);
				doc->words[i-1] = word_id;
				doc->counts[i-1] = count;
				total += count;
				if (max_word_id < word_id) {
					max_word_id = word_id;
				}
			}
			doc->total = total;
			docs.push_back(doc);
		}
		this->num_terms = max_word_id +1;
		fclose(file);

        cout << "REMOVEME - end read data. #docs = " << docs.size() << ".#terms = " << num_terms << endl;


	};
	int get_max_docs_length(){
		int max_ = 0;
		std::for_each(docs.begin(), docs.end(), [&max_](Document* doc) {
			if (max_ < doc->total) {
				max_ = doc->total;
			}
		});
		return max_;

	};

	void L1_normalize_document(){
		std::for_each(docs.begin(), docs.end(), [](Document* doc){
			for(int i=0; i < doc->length; i++){
				doc->counts[i] /= 1.0*doc->total;
			}
		});

	}
	~Corpus(){
		for(auto d: docs){
			delete d;
		}
	}
};

struct Model {
    Corpus* corpus;

    float EM_CONVERGED;
	int EM_MAX_ITER;
	int INF_MAX_ITER;
	float LAMBDA;		// penalty on Tr(inv_sigma)

    double** bb;			//topics
    int num_topics;
    int num_terms;

    double*  mu;		//Parameters for Gaussian
	double** inv_sigma;	//Parameters for Gaussian
	double* inv_sigma_sum;	//sum of rows of Inv_sigma
	double log_det_cov;
	double stat;		//-0.5 *(K-1) *log(2*pi) - 0.5*model->log_det_cov;

	double** aa;		// new presenation of document

	int n_threads;
	string output_dir;

public:
	Model(Corpus* corpus, int EM_MAX_ITER, double EM_CONVERGED, int INF_MAX_ITER,
		double LAMBDA, int num_topics, int n_threads, string out)
	{
        this->corpus = corpus;

		this->EM_MAX_ITER = EM_MAX_ITER;
		this->EM_CONVERGED = EM_CONVERGED;
		this->INF_MAX_ITER = INF_MAX_ITER;
		this->LAMBDA = LAMBDA;

        this->num_terms	     = corpus->num_terms;
        this->bb			 = initialize_matrix(num_topics, num_terms); //topics
        this->num_topics     = num_topics;

        this->mu			 = new double [num_topics-1];
        this->inv_sigma	 	 = initialize_matrix(num_topics -1, num_topics -1);
		this->inv_sigma_sum  = new double [num_topics -1];

        this->aa  			 = initialize_matrix(corpus->docs.size(), num_topics);
		 						//new presenation of documents
								// log theta

		this->n_threads		 = n_threads;
		this->output_dir 	 = out;
	}

	~Model(){
		for (int i = 0; i < num_topics; ++i)
    		delete [] this->bb[i];
		delete [] this->bb;

		for (size_t d = 0; d < corpus->docs.size(); ++d)
    		delete [] this->aa[d];
		delete [] this->aa;

		delete this->mu;
		delete this->inv_sigma_sum;

		for(int i = 0; i < num_topics -1; i++)
			delete this->inv_sigma[i];
		delete this->inv_sigma;

	}

	void learn(){

        cout << "Learning start...\n" ;
		string log_file = output_dir + "/" + "log.txt";
		FILE* fileptr = fopen(log_file.c_str(), "w");

		//run em
		double jointProb_old = 0, converge_joint = 1;
        double jointProb;

        initialize_M_step();
		for(size_t d = 0; d < corpus->docs.size(); d++){
			corpus->docs[d]->a = aa[d];
		}

        int i = 0;
		while (i < EM_MAX_ITER && converge_joint > EM_CONVERGED) {

            jointProb = this->stat;

			clock_t start = clock();

			Parallel::Parallel *pool = new Parallel::Parallel(n_threads);
			pool->foreach(corpus->docs.begin(),corpus->docs.end(), [&](Document* doc){
				doc_projection(doc);
		  	});
			for (auto d : corpus-> docs) {
				jointProb += d->obj;
			}
			converge_joint = fabs(jointProb - jointProb_old) / fabs(jointProb_old);
			jointProb_old  = jointProb;
			delete pool;

			clock_t stop = clock();
			double elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
			cout << "E-step time: " << elapsed << " s." << endl;


			start = clock();
			M_step();
			stop = clock();
			elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
			cout << "M-step time: " << elapsed << " s." << endl;


            i++; 	cout << "  **** iteration  "  << i << "jointProb: "<< jointProb << "****\n";
			fprintf(fileptr, "iter: %d - jointProb: %1.10f\n", i, jointProb);
			if ( i == 1 || (i > 5 && i%10 == 0) ) {
				save(i);
			}
		}

		fclose(fileptr);
        cout << "Learning END..." ;
	}

	void save(int iter) {
		cout << "Saving model at iter.." << iter<< '\n';
		FILE *fileptr;

		string beta_final = output_dir + "/" + "beta.iter_ " + std::to_string(iter);
		fileptr = fopen(beta_final.c_str(), "w");

		if (bb == NULL) {
			std::cout << "bb is nULL" << '\n';
		}
	    for (int i = 0; i < num_topics; i++) {
	        for (int j = 0; j < num_terms; j++) {
	            fprintf(fileptr, "%1.10f ", bb[i][j]);
	        }
	        fprintf(fileptr, "\n");
	    }
	    fclose(fileptr);


		string theta_final = output_dir + "/" + "theta.iter_" + std::to_string(iter);

		fileptr = fopen(theta_final.c_str(), "w");
	    for (auto doc : corpus->docs) {
			fprintf(fileptr, "%s\t", doc->doc_id.c_str());
	        for (int j = 0; j < num_topics; j++) {
	            fprintf(fileptr, "%1.10f ", doc->a[j]);
	        }
	        fprintf(fileptr, "\n");
	    }
	    fclose(fileptr);

		string muy_final = output_dir + "/" + "muy.iter_" + std::to_string(iter);
		fileptr = fopen(muy_final.c_str(), "w");
		for(int i = 0; i < num_topics-1; i++){
			fprintf(fileptr, "%1.10f ", mu[i]);
		}
	    fclose(fileptr);

		string inv_sigma_final = output_dir + "/" + "inv_sigma.iter_" + std::to_string(iter);
		fileptr = fopen(inv_sigma_final.c_str(), "w");
		for(int i = 0; i < num_topics-1; i++){
			for(int j = 0; j < num_topics-1; j++){
				fprintf(fileptr, "%1.10f ", inv_sigma[i][j]);
			}
			fprintf(fileptr, "\n");
		}
	    fclose(fileptr);

		string docs_lkh = output_dir + "/" + "docs_lkh.iter_" + std::to_string(iter);
		fileptr = fopen(docs_lkh.c_str(), "w");
		for (auto doc : corpus->docs) {
			fprintf(fileptr, "docID: %s - likelihood: %f\n", doc->doc_id.c_str(), doc->likelihood);
		}
		fclose(fileptr);


		std::cout << "Save model in " << output_dir <<'\n';
	}

	void M_step() {
		//update the topics: topic[k][j] ~= sum_{d} aa[k][d] * N[d][j]
		int i, j, k, KK;
		double trace;

		KK = num_topics-1;
		//compute Mu
		for (k = 0; k < KK; k++) 	{
			mu[k] = 0;
			for (size_t d = 0; d < corpus->docs.size(); d++) {
				mu[k] += aa[d][k] - aa[d][KK];
			}
			mu[k] /= corpus->docs.size();
		}
		//compute inv_sigma
		for (k = 0; k < KK; k++) {
			d_fill(inv_sigma[k], 0, KK);
		}
		for (k = 0; k < KK; k++) {
			for (i = k; i < KK; i++) {
				for (size_t d = 0; d < corpus->docs.size(); d++) {
					inv_sigma[k][i] += (aa[d][k] - aa[d][KK] - mu[k]) * (aa[d][i] - aa[d][KK] - mu[i]);
				}
			}
		}
		for (k = 0; k < KK; k++) {
			for (i = k; i < KK; i++) {
				inv_sigma[k][i] *= 1.0 / corpus->docs.size();
				inv_sigma[i][k] = inv_sigma[k][i];
			}
		}
		if (LAMBDA > 0) {
			for (k = 0; k < KK; k++) {
				inv_sigma[k][k] += LAMBDA;
			}
		}

		log_det_cov = det_inv_covariance(inv_sigma, KK);

		for (k = 0; k < KK; k++) {
			inv_sigma_sum[k] = 0;
			for (i = 0; i < KK; i++) {
				inv_sigma_sum[k] += inv_sigma[i][k];
			}
		}
		//recover topic proportion
		for (size_t d = 0; d < corpus->docs.size(); d++) {
			for (k = 0; k < num_topics; k++)	{
				aa[d][k] = exp(aa[d][k]);
			}
		}
		//compute topics
		for (k = 0; k < num_topics; k++) {
			d_fill(bb[k], 0, num_terms);
		}
		for (size_t d = 0; d < corpus->docs.size(); d++) {
			for (j = 0; j < corpus->docs[d]->length; j++) {
				for (k = 0; k < num_topics; k++) {
					bb[k][corpus->docs[d]->words[j]] += aa[d][k] * corpus->docs[d]->counts[j];
				}
			}
		}
		for (k = 0; k < num_topics; k++) {
			L1_normalize(bb[k], num_terms);
		}
		trace = 0;
		for (k = 0; k < KK; k++) {
			trace += inv_sigma[k][k];
		}
		trace *= LAMBDA;
        this->stat = -0.5 *(trace + log_det_cov) * corpus->docs.size();

		return;
	}

	double det_inv_covariance(double **A, int dim) {
		//The function computes inverse of A and stores inversion in A. It returns log[det(A)]
		gsl_matrix *evects, *mcpy;
		gsl_vector *evals;
		int k, i, j;
		double dete;
		gsl_eigen_symmv_workspace* wk;

		mcpy = gsl_matrix_alloc(dim, dim);
		evects = gsl_matrix_alloc(dim, dim);
		evals = gsl_vector_alloc(dim);
		wk = gsl_eigen_symmv_alloc(dim);

		//Eigen decomposition
		for (i=0; i<dim; i++) {
			for (j=0; j<dim; j++) {
				gsl_matrix_set(mcpy, i, j, A[i][j]);
			}
		}
		i = gsl_eigen_symmv(mcpy, evals, evects, wk);

		gsl_eigen_symmv_free(wk);
		gsl_matrix_free(mcpy);

		double eva[dim], **V;
		V = initialize_matrix(dim, dim);
		dete = 0;				//log[det(A)]
		for (i=0; i<dim; i++) {
			eva[i] = gsl_vector_get(evals, i);
			dete  += log(eva[i]);
			eva[i] = 1.0 / eva[i];
		}
		for (i=0; i<dim; i++) {
			for (j = 0; j<dim; j++)  {
				V[i][j] = gsl_matrix_get(evects, i, j);
			}
		}
		//eigenvectors --> LU
		for (i=0; i<dim; i++) {
			eva[i] = sqrt(eva[i]);
		}
		for (i=0; i<dim; i++) {
			for (j = 0; j<dim; j++)  {
				V[j][i] *= eva[i];
			}
		}
		//inverse =  V*V^t
		for (i=0; i<dim; i++) {
			for (j = i; j<dim; j++) {
				A[i][j] = 0;
				for (k = 0; k< dim; k++) {
					A[i][j] += V[i][k]*V[j][k];
				}
				A[j][i] = A[i][j];
			}
		}
		for (i = dim-1; i>=0; i--) {
			free(V[i]);
		}
		gsl_matrix_free(evects);
		gsl_vector_free(evals);
		return (dete);
	}


	void L1_normalize(double *vec, int dim) {
		int i;	double sum=0;
		for (i=0; i< dim; i++)	{
	        sum += vec[i];
	    }
		for (i=0; i< dim; i++)	{
	        vec[i] /= sum;
	    }
	}


    void initialize_M_step() {
        cout << "init M step " << endl;
		srand(time(0));

		int k, KK;
        initialize_random_topics();
        KK = num_topics -1;
        for (k = 0; k < KK; k++) {
            mu[k]  = 0;
        }
        //Diagonal Sigma
        for (k = 0; k < KK; k++) {
            d_fill(inv_sigma[k], 0, KK);
        }

        for (k = 0; k < KK; k++) {
            inv_sigma[k][k] = 1;
            inv_sigma_sum[k] = 1;
        }
        log_det_cov = 0;
        //for (k = 0; k < KK; k++) model->log_det_cov -= log(model->inv_sigma[k][k]);
        stat = -0.5 *(KK + log_det_cov) * corpus->docs.size();

        cout << "End init M step" << endl;
        return;

    }

    void initialize_random_topics() {
        cout << "initialize_random_topics" << endl;
        for (int i = 0; i < num_topics; i++) {
            for (int j = 0; j < num_terms; j++) {
                bb[i][j] = rand() +1;
            }
            L1_normalize(bb[i], num_terms);
        }
        cout << "end initialize_random_topics" << endl;
        return ;
    }

    double** initialize_matrix(int rows, int cols) {
        //initialize a matrix randomly
        double **aa = new double*[rows];
        for (int i = 0; i < rows; i++) {
            aa[i] = new double[cols];
        }
        return (aa);
    }

    void d_fill(double *a, double value, int dim) {
        for (int i=0; i < dim; i++) {
            a[i] = value;
        }
    }

///  -------------- for E_step
    //Objective function is the joint P(d, theta)
    double f_joint(Document *doc, double *theta, double *x) {
    	// compute f(theta).		x = beta * theta
    	int j, k, KK;
    	double sum, fx=0;

    	for (j = 0; j < doc->length; j++) {
    		fx += doc->counts[j] * log(x[j]);
    	}
    	KK = num_topics -1;
    	for (k = 0; k < KK; k++){
    		sum = 0;
    		for (j = 0; j < KK; j++) {
    			sum += (theta[j] - theta[KK] - mu[j]) * inv_sigma[j][k];
    		}
    		fx -= 0.5 *sum *(theta[k] - theta[KK] - mu[k]);
    	}
    	for (k = 0; k < KK+1; k++) {
    		fx -= theta[k];
    	}
    	return (fx);
    }

    double df_joint(Document *doc, double *theta, double *x, int ind, double* T) {
    	// compute dfx(theta)_ind.		x = beta * theta
    	int  j, KK;
    	double sum, dfx=0;
    	for (j = 0; j < doc->length; j++) {
    		dfx += doc->counts[j] * bb[ind][doc->words[j]] / x[j];
    	}
    	dfx *= T[0];
    	KK =num_topics -1;
    	sum = -1;
    	if (ind < KK) {
    		for (j = 0; j < KK; j++) {
    			sum -= inv_sigma[ind][j] * (theta[j] -theta[KK] - mu[j]);
    		}
    	} else {
    		for (j = 0; j < KK; j++) {
    			sum += inv_sigma_sum[j] * (theta[j] -theta[KK] - mu[j]);
    		}
    	}
    	dfx += T[1] * (sum / exp(theta[ind]));
    	return (dfx);
    }

    //Objective function is the Likehood
    double f_lkh(Document *doc, double *theta, double *x) {
    	// compute f(theta).		x = sum_k {beta_k * theta_k}
    	double fx=0;
    	for (int j = 0; j < doc->length; j++) {
    		fx += doc->counts[j] * log(x[j]);
    	}
    	return (fx);
    }

    double df_lkh(Document *doc, double *theta, double *x, int ind) {
    	// compute dfx(theta)_ind.		x = sum_k {beta_k * theta_k}
    	double dfx=0;
    	for (int j = 0; j < doc->length; j++) {
    		dfx += doc->counts[j] * bb[ind][doc->words[j]] / x[j];
    	}
    	return (dfx);
    }

    void doc_projection(Document *doc) {
		//Maximizing f(x) over the simplex of topics, using Online Frank-Wolfe
		double obj, obj_max, fmax, alpha, *opt, sum, EPS;
		int i, t, ind, no_improvement=0;

		opt = new double [doc->length]; //opt_j = sum_k {theta_k * beta_kj}
		double *theta = doc->a;


		EPS   = log(1e-10);
		alpha = log(1 - (num_topics-1)*exp(EPS));
		for (i = 0; i < num_topics; i++) {
			theta[i] = EPS;
		}


		std::random_device rand_dev;
	    std::mt19937 generator(rand_dev());
	    std::uniform_int_distribution<int>  distr_topic(0,num_topics-1);

	    ind = distr_topic(generator);
		theta[ind] = alpha;

		for (i = 0; i < doc->length; i++) {
			opt[i] = bb[ind][doc->words[i]];
		}

	    //online Frank Wolfe
        double T[2]; T[0] = 1; T[1] = 0;
		std::uniform_int_distribution<int>  distr_T(0,1);

		for (t = 1; t < INF_MAX_ITER; t++) {
			T[distr_T(generator)]++;	//pick a part of the objective function (0/1)
			ind = -1;	//select the best direction
			for (i = 0; i < num_topics; i++) {
				sum = df_joint(doc, theta, opt, i, T);
				if (ind < 0 || sum > fmax)	{
					fmax = sum;	ind = i;
				}
			}
			alpha = 2.0 / (t + 2);
			for (i = 0; i < num_topics; i++) {
				theta[i] += log(1-alpha);
			}
			theta[ind] = log( exp(theta[ind]) + alpha );
			for (i = 0; i < doc->length; i++) {
				opt[i] = (1-alpha) * opt[i] + alpha * bb[ind][doc->words[i]];
			}

		}

		for (i = 0; i < doc->length; i++) {
			opt[i] = 0;
			for (t = 0; t < num_topics; t++) {
				opt[i] += exp(theta[t]) * bb[t][doc->words[i]];
			}
		}
		//Compute objective value and likelihood
		double lkh = 0;
		for (i = 0; i < doc->length; i++) {
			lkh += doc->counts[i] * log(opt[i]);
		}

		obj = f_joint(doc, theta, opt);

		doc->obj = obj;
		doc->likelihood = lkh;

		delete opt;
		return;
	}

};


int main(int argc, char* argv[]) {
    std::cout << "x" << std::endl;
    std::ifstream ifs("CTM.toml");
    toml::ParseResult pr = toml::parse(ifs);

    if (!pr.valid()) {
        std::cout << pr.errorReason << std::endl;
    return 1;
    }

    string location_data = pr.value.get<string>("CTM.location_data");
    int EM_MAX_ITER =  pr.value.get<int>("CTM.EM_MAX_ITER");
    double EM_CONVERGED = pr.value.get<double>("CTM.EM_CONVERGED");
    int INF_MAX_ITER = pr.value.get<int>("CTM.INF_MAX_ITER");
    double LAMBDA =  pr.value.get<double>("CTM.LAMBDA");
    int threads = pr.value.get<int>("CTM.threads");
    int num_topics =  pr.value.get<int>("CTM.n_topics");

	string output_dir = pr.value.get<string>("CTM.output_dir");

    Corpus* corpus = new Corpus();
    corpus->read_data(location_data);

    Model* model = new Model(corpus, EM_MAX_ITER, EM_CONVERGED, INF_MAX_ITER, LAMBDA, num_topics, threads, output_dir);
    model->learn();

    delete model;
    delete corpus;
    return 0;
}
