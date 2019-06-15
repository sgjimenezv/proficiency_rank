import codecs
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import sys
import pickle
from nltk.tokenize import word_tokenize

print("Reading CEFR word lists ...")
input_file=open("wordlists/CEFR_WORDLISTS.pkl","rb")
CEFR=pickle.load(input_file)
input_file.close()
print("Done!")
CEFR_levels={"C2":6,"C1":5,"B2":4,"B1":3,"A2":2,"A1":1}
#CEFR_levels={"C2":1,"C1":2,"B2":3,"B1":4,"A2":5,"A1":6}

if len(sys.argv)>=2:
	damping=float(sys.argv[1])
else:
	damping=0.95

if len(sys.argv)>=3:
	negative_weight=float(sys.argv[2])
else:
	negative_weight=0.8
	
if len(sys.argv)>=4:
	pool_weight_neg=float(sys.argv[3])
else:
	pool_weight_neg=0.2

if len(sys.argv)>=5:
	pool_weight_pos=float(sys.argv[4])
else:
	pool_weight_pos=0.4

	
print("damping=",damping,"negative_weight=",negative_weight,"pool_weight_neg",pool_weight_neg,pool_weight_pos)	
	
# Parameter M adjacency matrix where M_i,j represents the link from 'j' to 'i', such that for all 'j'
# sum(i, M_i,j) = 1
# Parameter d damping factor (ddefault value 0.85)
# Parameter eps quadratic error for v (default value 1.0e-8)
# Return v, a vector of ranks such that v_i is the i-th rank from [0, 1]

import numpy as np
def pagerank(M, eps=1.0e-8, d=0.85):
    N = M.shape[1]
    v = np.random.rand(N)
    #v=np.array((0.5,0.5,0.5,0.5,0.5))
    #print(v)
    v = v / np.linalg.norm(v, 1)
    #print("v1",v)
    #print("v1.sum",v.sum())
    last_v = np.ones((N, 1), dtype=np.float32) * 100
    #print("last_v",np.ones((N, 1), dtype=np.float32) * 100)
    M_hat = (d * M) + (((1 - d) / N) * np.ones((N, N), dtype=np.float32))
    #print("M_hat",M_hat)
    i=0
    while np.linalg.norm(v - last_v, 2) > eps:
        last_v = v
        v = np.matmul(M_hat, v)
        i+=1
        #print(i,"v",v)
        #print(i,"v.sum",v.sum())
        #if i>2:
	    #    break
    return v







	
def proficiency_rank(M_exp_pos,M_exp_neg,M_aiv,M_oiv,log=False,ds=[0.95,],alphas=[0.8,],betas=[0.0,],deltas=[0.0,],baseline=None):

	def normalize_cols(matrix):
	#normalize columns to sum up 1
		s=matrix.sum(axis=0)
		d=(s==0)
		s[d]=1
		matrix=matrix/s
		# replaces columns with 1/n
		#print(d.shape)
		tmp=np.ones(matrix.shape)*1/matrix.shape[0]
		matrix[:,d]=tmp[:,d]
		return(matrix)
		
	
	incoming_votes=np.count_nonzero(M_exp_pos,axis=1)
	incoming_votes+=np.count_nonzero(M_exp_neg,axis=1)
	incoming_votes+=np.count_nonzero(M_aiv,axis=1)
	incoming_votes+=np.count_nonzero(M_oiv,axis=1)
	
	incoming_votes_weights=np.sum(M_exp_pos,axis=1)
	incoming_votes_weights+=np.sum(M_exp_neg,axis=1)
	incoming_votes_weights+=np.sum(M_aiv,axis=1)
	incoming_votes_weights+=np.sum(M_oiv,axis=1)
	
	
	#incoming_votes=M.sum(axis=1)+M_.sum(axis=1)+M1.sum(axis=1)+M2.sum(axis=1)
	print("# Explicit positive votes",np.count_nonzero(M_exp_pos,axis=1).sum())
	print("# Explicit negative votes",np.count_nonzero(M_exp_neg,axis=1).sum())
	print("# Implicit agreement votes",np.count_nonzero(M_aiv,axis=1).sum())
	print("# Implicit opposition votes",np.count_nonzero(M_oiv,axis=1).sum())	
	print("Total Votes:",incoming_votes.sum())
	print()
	print("d alpha beta delta best_r p #users #votes first_r p #users")
	for damping in ds:
		for alpha in alphas:
			for beta in betas:
				for delta in deltas:
					
					print("d=",round(damping,2),"alpha=",round(alpha,2),"beta=",round(beta,2),"delta=",round(delta,2),end=" ")
					M_pos=normalize_cols((1-beta)*M_exp_pos+beta*M_aiv)
					M_neg=normalize_cols((1-delta)*M_exp_neg+delta*M_oiv)
					R_pos=pagerank(M_pos,d=damping)
					R_neg=pagerank(M_neg,d=damping)
					R=(1-alpha)*R_pos-alpha*R_neg
					
					#evaluation
					#first_correlation=(0,0) #correlation,p_value, #users
					best_correlation=(-999,0,0,0) #correlation, p_value, #users, min_votes
					sum_correlation=0.0
					n_correlation=0
					
					#baseline computation incoming_votes vs. goldstandard
					baseline_correlation,baseline_p_value=spearmanr(incoming_votes,[users[users_names[i]][1] for i in range(len(incoming_votes))])
					print("\nBaseline1:",baseline_correlation,baseline_p_value,len(incoming_votes))
					baseline_correlation,baseline_p_value=spearmanr(incoming_votes_weights,[users[users_names[i]][1] for i in range(len(incoming_votes_weights))])
					print("\nBaseline2:",baseline_correlation,baseline_p_value,len(incoming_votes))
					for min_votes in range(1,int(incoming_votes.max()-1)):
						pr=[]# proficiency rank
						gs=[]# gold standard
						bl=[]# baseline
						for i in range(len(R)):
							if incoming_votes[i]>=min_votes:
								pr.append(R[i])
								gs.append(users[users_names[i]][1])
								#bl.append(incoming_votes[i])
								bl.append(baseline[i])
						correlation=spearmanr(pr,gs)[0]
						p_value=spearmanr(pr,gs)[1]
						#if min_votes==1:
						#	first_correlation=(correlation,p_value,len(pr))
						if p_value<0.01 and len(pr)>10:
							if abs(correlation)>best_correlation[0]:
								best_correlation=(correlation,p_value,len(pr),min_votes)
							if log :
								print("\t",min_votes,len(pr),round(spearmanr(pr,gs)[0],6),round(spearmanr(pr,gs)[1],6),"Baseline:",round(spearmanr(bl,gs)[0],6),round(spearmanr(bl,gs)[1],6) )
							sum_correlation+=correlation
							n_correlation+=1
					#print("r=(",round(best_correlation[0],6),round(best_correlation[1],6),") min_votes=",best_correlation[3],"users=",best_correlation[2],")(",round(first_correlation[0],6),round(first_correlation[1],6),first_correlation[2],")")
					if n_correlation>0:
						print("Avg.:",round(sum_correlation/n_correlation,6),n_correlation,round(best_correlation[0],6),best_correlation[2],best_correlation[3])
					else:
						print()




					
levels={"Nativo":5,"Fluido":4,"Avanzado":3,"Intermedio":2,"Principiante":1}
#levels={"Nativo":7,"Fluido":6,"Avanzado":5,"Intermedio":3,"Principiante":1}
distribution={"Nativo":0,"Fluido":0,"Avanzado":0,"Intermedio":0,"Principiante":0}
users={}
users_names={}
yeah_votes=[]
nay_votes=[]
pool_votes_neg=[]
pool_votes_pos=[]

# reads users.tsv file

print("Reading users.tsv file ...")
i=0
id_user=0
input_file=codecs.open("en_users.tsv","r","utf-8")

for line in input_file:
	i+=1
	if i==1:
		continue #skips header line
	fields=line.split("\t")
	#print(fields)
	#language=fields[4]
	if fields[0]!="":
		user_name=fields[0].strip()
#	if user_name in users_to_remove:
#		continue
	level=fields[1]
	#if language!="Inglés" or user_name=="":
	#	continue #skips non English lines
	#print(user_name,level)
	if user_name in users:
		print("Duplicated user",user_name,"line",i,language)
		continue
	else:
		users[user_name]=(id_user,levels[level])
		users_names[id_user]=user_name
		id_user+=1
		distribution[level]+=1
print("Number of users",len(users),len(users_names),i,"lines", id_user)
print(distribution)

#reads votes
print("Reading votes.tsv file")
i=0
input_file=codecs.open("en_votes.tsv","r","utf-8")

oiv_pos=[]
oiv_neg=[]
aiv_pos=[]
aiv_neg=[]
positive_pool=[]
negative_pool=[]
last_answer=""
requests=[]
answers=[]
answers_users={}
for line in input_file:
	i+=1
	if i==1:
		continue #skips headers line
	fields=line.split("\t")
	if fields[1]!="":
		requests.append(fields[1])
	if fields[4]!="":
		answer=fields[4]
	if fields[3]!="":
		lang_text=fields[3]
	if fields[2]!="":
		request_text=fields[2]
	if fields[5]!="":
		lang_answer=fields[5]
	if fields[6]!="":
		author_answer=fields[6].strip()
		answers.append(author_answer)
	yeah_user=fields[7].strip()
	nay_user=fields[8].strip()
	
	#only English filter
	if not (lang_text=="EN" or lang_answer=="EN"):
		continue
	
	# records user answers
	if fields[4]!="" and fields[6]!="":
		if lang_answer=="EN":
			answer=fields[4].strip()
		else:
			answer=request_text
		if fields[6] in answers_users:
			answers_users[fields[6]].append(answer)
		else:
			answers_users[fields[6]]=[answer]
	#
	if answer==last_answer:
		if yeah_user!="":
			positive_pool.append(yeah_user)
		if nay_user!="":
			negative_pool.append(nay_user)
	else:
		#OPOSITE IMPLICIT VOTES
		if len(negative_pool)>0:
			for negative_voter in negative_pool:
				for positive_voter in positive_pool:
					oiv_neg.append((negative_voter,positive_voter))
					oiv_pos.append((positive_voter,negative_voter))
		#AGREEMENT IMPLICIT VOTES
		for positive_voter1 in positive_pool:
			for positive_voter2 in positive_pool:
				if positive_voter1!=positive_voter2:
					aiv_pos.append((positive_voter1,positive_voter2))
		for negative_voter1 in negative_pool:
			for negative_voter2 in negative_pool:
				if negative_voter1!=negative_voter2:
					aiv_neg.append((negative_voter1,negative_voter2))

		last_answer=answer
		positive_pool=[]
		negative_pool=[]
		
		

	# check users
	if not author_answer in users: 
		print("Author anwser user not found in line",i,author_answer,author_answer in users)
		continue
	if  yeah_user!="" and not yeah_user in users:
		print("Yeah user not found in line",i,yeah_user)
		continue
	if nay_user!="" and not nay_user in users:	
		print("Nay user not found in line",i,nay_user)
		continue
	if yeah_user!="":
		yeah_votes.append((yeah_user,author_answer))
	if nay_user!="":
		nay_votes.append((nay_user,author_answer))
input_file.close()
print("yeah_votes",len(yeah_votes)) 
print("nay_votes",len(nay_votes)) 
print("agreement implicit votes (pos)",len(aiv_pos)) 
print("agreement implicit votes (neg)",len(aiv_neg))
print("opposition implicit votes (neg)",len(oiv_pos)) 
print("opposition implicit votes (pos)",len(oiv_neg))
 
#sys.exit()


# builds the positive matrix
M_EXP_POS=np.zeros((len(users),len(users)))
for vote_from, vote_to in yeah_votes:
	M_EXP_POS[users[vote_to][0],users[vote_from][0]]+=1

M_AIV_POS=np.zeros((len(users),len(users)))
for vote_from, vote_to in aiv_pos:
	M_AIV_POS[users[vote_to][0],users[vote_from][0]]+=1

M_AIV_NEG=np.zeros((len(users),len(users)))
for vote_from, vote_to in aiv_neg:
	M_AIV_NEG[users[vote_to][0],users[vote_from][0]]+=1


# builds the matrix negative
M_EXP_NEG=np.zeros((len(users),len(users)))
for vote_from, vote_to in nay_votes:
	M_EXP_NEG[users[vote_to][0],users[vote_from][0]]+=1
	
M_OIV_POS=np.zeros((len(users),len(users)))
for vote_from, vote_to in oiv_pos:
	M_OIV_POS[users[vote_to][0],users[vote_from][0]]+=1

M_OIV_NEG=np.zeros((len(users),len(users)))
for vote_from, vote_to in oiv_neg:
	M_OIV_NEG[users[vote_to][0],users[vote_from][0]]+=1

ZEROS=np.zeros((len(users),len(users)))





# outputs the answers from users
print("Saving user answers in answers_users.tsv")
output_file=codecs.open("answers_users.tsv","w","utf-8")
output_file.write("USER\tLEVEL\tANSWER\r\n")
sorted_users=[users[u][0] for u in answers_users]
sorted_users.sort()
CEFR_baseline={}
#print(sorted_users)
for id_user in sorted_users:
	user=users_names[id_user]
	level=users[user][1]
	answers=answers_users[user]
	#(id_user,level)=users[user]
	words=[]
	for answer in answers:
		output_file.write("{0}\t{1}\t{2}\r\n".format(id_user,level,answer))
		tokens=word_tokenize(answer)
		for token in tokens:
			if token.isalpha():
				words.append(token.lower())
	words=set(words)
	print(id_user,end=" ")
	weighted_sum=0.0
	summation=0.0
	max_value=0
	max_level=0
	for cefr_level in ["A1","A2","B1","B2","C1","C2",]:
		print(cefr_level,len(CEFR[cefr_level].intersection(words)),end="  ")
		weighted_sum+=CEFR_levels[cefr_level]*len(CEFR[cefr_level].intersection(words))
		summation+=len(CEFR[cefr_level].intersection(words))
		if len(CEFR[cefr_level].intersection(words))>=max_value:
			max_value=len(CEFR[cefr_level].intersection(words))
			max_level=CEFR_levels[cefr_level]
	print(weighted_sum/summation,max_level)
	CEFR_baseline[id_user]=weighted_sum/summation
	#CEFR_baseline[id_user]=max_level
output_file.close()
#sys.exit()



#CONFIGURATION #1	
#proficiency_rank(M_EXP_POS,M_EXP_NEG,ZEROS,ZEROS,ds=[0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.90],alphas=[0.75,0.76,0.77,0.78,0.79,0.80,0.81,0.82,0.83,0.84,0.85])
#d= 0.86 alpha= 0.79 beta= 0.0 delta= 0.0 Avg.	 0.676543 33
#d=	0.86	alpha=	0.79	beta=	0	delta=	0	Avg.:	0.676543	33	0.752705	12	31
#d=	0.87	alpha=	0.79	beta=	0	delta=	0	Avg.:	0.676495	33	0.752705	12	31
#d=	0.87	alpha=	0.8	beta=	0	delta=	0	Avg.:	0.676375	33	0.75678	26	16
#d=	0.89	alpha=	0.79	beta=	0	delta=	0	Avg.:	0.676283	33	0.752705	12	31
print("\n\n\nConfiguration #1")
proficiency_rank(M_EXP_POS,M_EXP_NEG,ZEROS,ZEROS,log=True,ds=[0.86,],alphas=[0.79,],baseline=CEFR_baseline)
sys.exit()



#CONFIGURATION #2
#proficiency_rank(M_EXP_POS,M_EXP_NEG,M_AIV_NEG,M_OIV_NEG,
#ds=[0.80,0.85,0.90],
#alphas=[0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79],
#betas=[0.5,0.6,0.7,0.8,0.9,1.0,],
#deltas=[0.4,0.45,0.50,0.55,0.60],)
#d=	0.8	alpha=	0.78	beta=	0.9	delta=	0.4	Avg.:	0.609286	48	0.779895	17	37
#d=	0.8	alpha=	0.78	beta=	0.9	delta=	0.4	Avg.:	0.609286	48	0.779895	17	37
print("\n\n\nConfiguration #2")
proficiency_rank(M_EXP_POS,M_EXP_NEG,M_AIV_NEG,M_OIV_NEG,log=True,ds=[0.8,],alphas=[0.78,],betas=[0.9,],deltas=[0.4])

#sys.exit()

#CONFIGURATION #3
#proficiency_rank(M_EXP_POS,M_EXP_NEG,ZEROS,M_OIV_POS+M_OIV_NEG,
#ds=[0.84,0.85,0.86],
#alphas=[0.84,0.85,0.86],
#betas=[0,],
#deltas=[0.14,0.15,0.16],)
#d=	0.85	alpha=	0.85	beta=	0	delta=	0.15	Avg.:	0.606324	42	0.779896	17	31
print("\n\n\nConfiguration #3")
proficiency_rank(M_EXP_POS,M_EXP_NEG,ZEROS,M_OIV_POS+M_OIV_NEG,log=True,ds=[0.85,],alphas=[0.85,],betas=[0.0,],deltas=[0.15])


#CONFIGURATION #4
#proficiency_rank(M_EXP_POS,M_EXP_NEG,M_AIV_POS,M_OIV_POS,
#ds=[0.97,0.98,0.99],
#alphas=[0.38,0.39,0.40,],
#betas=[0.39,0.4,0.41,],
#deltas=[0.73,0.74,0.75],)
#d=	0.98	alpha=	0.39	beta=	0.4	delta=	0.74	Avg.:	0.401515	34	0.603916	21	62
print("\n\n\nConfiguration #4")
proficiency_rank(M_EXP_POS,M_EXP_NEG,M_AIV_POS,M_OIV_POS,log=True,ds=[0.98,],alphas=[0.39,],betas=[0.4,],deltas=[0.74])



#CONFIGURATION #5
#proficiency_rank(M_EXP_POS,M_EXP_NEG,M_AIV_POS+M_AIV_NEG,ZEROS,
#ds=[0.89,0.9,0.91,],
#alphas=[0.64,0.65,0.66,],
#betas=[0.09,0.1,0.11,],
#deltas=[0,],)
#d=	0.9	alpha=	0.65	beta=	0.1	delta=	0	Avg.:	0.530639	70	0.86088	12	74
print("\n\n\nConfiguration #5")
proficiency_rank(M_EXP_POS,M_EXP_NEG,M_AIV_POS+M_AIV_NEG,ZEROS,log=True,ds=[0.9,],alphas=[0.65,],betas=[.1,],deltas=[0.0])





#EXPERIMENT #6
#proficiency_rank(M_EXP_POS,M_EXP_NEG,M_AIV_POS+M_AIV_NEG,M_OIV_POS+M_OIV_NEG,
#ds=[0.84,0.85,0.86,],
#alphas=[0.66,0.67,0.68],
#betas=[0.13,0.14,0.15],
#deltas=[0.14,0.15,0.16],)
#d=	0.85	alpha=	0.66	beta=	0.14	delta=	0.15	Avg.:	0.428115	67	0.607567	28	61
print("\n\n\nConfiguration #6")
proficiency_rank(M_EXP_POS,M_EXP_NEG,M_AIV_POS+M_AIV_NEG,M_OIV_POS+M_OIV_NEG,log=True,ds=[0.85,],alphas=[0.66,],betas=[.14,],deltas=[0.15])
#sys.exit()




#EXPERIMENT #7
#proficiency_rank(M_EXP_POS,M_EXP_NEG,M_AIV_NEG,M_OIV_POS+M_OIV_NEG,
#ds=[0.88,0.89,0.90,.91],
#alphas=[0.84,0.85,0.86,],
#betas=[0.52,0.53,0.54,],
#deltas=[0.19,0.2,0.21,],)
#d=	0.89	alpha=	0.85	beta=	0.53	delta=	0.2	Avg.:	0.661514	62	0.793238	14	49
print("\n\n\nConfiguration #7")
proficiency_rank(M_EXP_POS,M_EXP_NEG,M_AIV_NEG,M_OIV_POS+M_OIV_NEG,log=True,ds=[0.89,],alphas=[0.85,],betas=[0.53,],deltas=[0.2,])
sys.exit()


#EXPERIMENT #8 ONLY POSITIVE VOTES
print("Only positive votes")
proficiency_rank(M_EXP_POS,ZEROS,ZEROS,ZEROS,log=True,ds=[0.4,0.5,0.6,0.7,0.8,0.9,],alphas=[0.0,],betas=[0.0,],deltas=[0.0,])
print("Only negative votes")
proficiency_rank(ZEROS,M_EXP_POS,ZEROS,ZEROS,log=True,ds=[0.4,0.5,0.6,0.7,0.8,0.9,],alphas=[1.0,],betas=[0.0,],deltas=[0.0,])

sys.exit()	
























#EXPERIMENT #7
#proficiency_rank(ZEROS,M_EXP_NEG,M_AIV_NEG,M_OIV_NEG,
#ds=[0.78,0.78,0.80],
#alphas=[0.42,0.43,0.44,0.45],
#betas=[1,],
#deltas=[0.59,0.60,0.61,],)
#d=	0.78	alpha=	0.43	beta=	1	delta=	0.6	Avg.:	0.613509	33	0.744008	14	26
#proficiency_rank(ZEROS,M_EXP_NEG,M_AIV_NEG,M_OIV_NEG,log=True,ds=[0.78,],alphas=[0.43,],betas=[1.0,],deltas=[0.6])





if False:
	j=0
	output_file=codecs.open("additional_output.txt","w","utf-8")
	for i in range(len(incoming_votes)):
		if incoming_votes[i]+outcoming_votes[i]==0.0:
			j+=1
			print(j,users_names[i],incoming_votes[i],outcoming_votes[i])
			output_file.write("'"+users_names[i]+"',\r\n")

	output_file.write("REQUESTS\r\n")
	for request in requests:

		try: 
			output_file.write(request+"-"+str(users[request][1])+"\r\n")
		except:
			pass
	answers.sort()
	output_file.write("\r\nANSWERS\r\n")
	answer_histogram={}
	for answer in answers:
		if answer in answer_histogram:
			answer_histogram[answer]+=1
		else:
			answer_histogram[answer]=1
		try:
			output_file.write(answer+"\r\n")
		except:
			pass
	answer_histogram=[(answer_histogram[a],a) for a in answer_histogram]
	answer_histogram.sort()

	for count,answer in answer_histogram:
		output_file.write(answer+":"+str(count)+"\r\n")
	output_file.write("# ANSWERS:"+str(len(answers))+"\r\n")
	output_file.write("# AUTHORS MAKEIN ANSWERS:"+str(len(set(answers)))+"\r\n")

	output_file.close()
