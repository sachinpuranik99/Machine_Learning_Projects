/* Code written by Sachin Shreenivas Puranik 
 * UIN: 427000135 */
/* Code implements a VPC indirect branch predictor which uses a hashed
 * perceptron predictor {mi,AxG,0} for the conditional prediction */

/* Hashed perceptron params */
#define NUMBER_OF_HISTORY_TABLES 4
//#define WEIGHTS_PER_TABLE 131072*16
#define WEIGHTS_PER_TABLE 8096*128
#define HIGHEST_WEIGHT_VALUE 127
#define LEAST_WEIGHT_VALUE -128
#define SHIFT_BITS 15
#define MASK_VALUE ((1<<SHIFT_BITS) - 1)
#define THETA (int)(1.93*NUMBER_OF_HISTORY_TABLES)+(NUMBER_OF_HISTORY_TABLES/2)

/* VPC params */
#define TAKEN 1 
#define NOT_TAKEN 0 

#define ARRAY 1

/* VPC params */
#ifdef ARRAY
#define MAX_ITER 52
#define BTB_SIZE ((1 << 27) - 1) 
#endif

#ifdef HASH
#define MAX_ITER 52
#define BTB_SIZE ((1 << 28) - 1) 
#endif

/* This class implements the hashed perceptron preditor that determines the conditional direction */
class hashed_perceptron 
{
public:
	bool prediction;
  int out;
	unsigned int index[NUMBER_OF_HISTORY_TABLES+0];
	unsigned long long int history_segment;
  unsigned long long int final_history_segment;
  int W[NUMBER_OF_HISTORY_TABLES+0][WEIGHTS_PER_TABLE];
  hashed_perceptron();
  bool predict(unsigned long long int); 
  void update(unsigned long long int, bool);
};

/* Constructor to reset stuff */
hashed_perceptron::hashed_perceptron()
{
		memset (index, 0, sizeof (index));
		memset (W, 0, sizeof (W));
    history_segment = 0;
    out =0;
    prediction = false;
    final_history_segment = 0;
}
	
/* Function to predict outcome for branch at PC */
bool hashed_perceptron::predict( unsigned long long int PC)
{
    index[0] = PC % WEIGHTS_PER_TABLE;
    out = W[0][index[0]];

    for(int j=1;j<NUMBER_OF_HISTORY_TABLES;j++)
    {
				final_history_segment = (((history_segment)) & (MASK_VALUE << SHIFT_BITS*(j-1))) >> SHIFT_BITS*(j-1);
        index[j] = (final_history_segment ^ PC) % (WEIGHTS_PER_TABLE);
				out += W[j][index[j]];	
    }
     
    if(out >= 0)
    {
			prediction = true;
    }
    else
    {
			prediction = false;
    }
    return prediction;
}

/* Function to train the predictor after the outcome for branch at PC is known */
void hashed_perceptron::update(unsigned long long int PC, bool outcome)
{
  /* To get the prediction value again, in case VPCAs are using it */
	  	bool local_prediction = this->predict(PC);  

      int absolute_out = (this->out >= 0)?this->out:((this->out)*-1); 

      if((local_prediction != outcome) || (absolute_out < THETA))
      {
        for(int j=0; j<NUMBER_OF_HISTORY_TABLES;j++)
        {
          if(outcome == true)
          {
            W[j][index[j]] = (W[j][index[j]] < HIGHEST_WEIGHT_VALUE)?(W[j][index[j]] + 1):W[j][index[j]];
          }
          else
          {
            W[j][index[j]] = (W[j][index[j]] > LEAST_WEIGHT_VALUE)?(W[j][index[j]] - 1):W[j][index[j]];
          }
        }
      }

      /* Updating the history_segment */
			history_segment <<= 1;
			history_segment|= outcome;
			history_segment&= 0xFFFFFFFFFFFFFFFF;
}

typedef struct BTB_struct {
  long int count;
  unsigned long int target;
  unsigned long int vpca;
  BTB_struct *next_node;
}BTB_struct;

/* Hash values */
static const long int VPC_CONSTANT[60]={0xfeeddead,0xdeadbeef,0xabdeadbe,0xe25389ac,0xe28f59fe,0x9d4fadf9,0x5a4ad01d,0x7bea7c56,0x5404f083,0xf1b9d562,0x86151511,0xef782057,0x78346abc,0xaa6b8f40,0x94e0117d,0x7cca0a1b,0x41912196,0x6f74fed5,0x3fba3222,0x54875dda,0x2da252fb,0x22a7ee7a,0xff24516a,0xfa29cec0,0xa5a55a5a,0xf9e8b71a,0xe8b71ade,0xf1e3b71b,0xe8b7e2ff,0xabcd1234,0xaaaa431a,0xffff555a,0xeeee1321,0xddd112ef,0xcccc449d,0xa1459021,0xf8829192,0x1b8ce9d8,0x254c15ec,0x21e679f8,0x23560d71,0x11054b73,0x3ab12d9f,0xf9e73553,0x25801db6,0xd994c167,0x1786b89c,0x35397dfa,0x38a33ec5,0xf089a35c,0x2c853b52,0x38ad77f9,0x100f0b76,0xfd6c74f3,0x20fcd43e,0x1ef148ee,0x23024098,0x26f7b600,0x3669d073,0xfd786d20};


/* This class implements the VPC preditor that determines the indirect prediction */
class vpc_predictor 
{
public:
  unsigned int VPCA; /* VPC */
  unsigned int pred_target; /* VPC */
  unsigned long long int VGHR; /* VPC */
  int iter; /* VPC */
  int predicted_iter; /* VPC */
  bool done; /* VPC */
  bool pred_dir; /* VPC */
  bool found_correct_target; /* VPC */
  BTB_struct* BTB; /* VPC */
  vpc_predictor();
#ifdef ARRAY
	unsigned int targets[BTB_SIZE + 1];
#endif
  unsigned long int predict(unsigned long int); 
  void update_algo1(unsigned long int);
  void update_algo2(unsigned long int, unsigned long int );
  unsigned long int access_BTB(unsigned long int);
  void update_replacement_BTB(unsigned long int);
  void insert_BTB(unsigned long int, unsigned long int);
};

/* Constructor to reset stuff */
vpc_predictor::vpc_predictor()
{
    VPCA = 0x0;
    pred_target = 0x0;
    VGHR = 0x0;
    iter = 0;
    predicted_iter = 0;
    done = false;
    pred_dir = false;
    found_correct_target=false; /* VPC */
    BTB = NULL;
#ifdef ARRAY
		memset (targets, 0, sizeof (targets));
#endif
}

hashed_perceptron hash_tron;
hashed_perceptron hash_tron1;
vpc_predictor vpc_pred;

/* Function to access the BTB and manage it */
unsigned long int vpc_predictor::access_BTB(unsigned long int PC)
{
#ifdef HASH
   BTB_struct* current_block = BTB;  // Initialize current
   /* Search if the PC exists in the BTB list */
    while (current_block != NULL)
    {
        if (current_block->vpca  == (PC & BTB_SIZE))
        {
          return current_block->target;
        }
        current_block= current_block->next_node;
    }
#endif

#ifdef ARRAY
    return targets[PC & BTB_SIZE];
#endif

    return 0;
}

void vpc_predictor::update_replacement_BTB(unsigned long int VPCA)
{
#ifdef HASH
   BTB_struct* current_block = BTB;  // Initialize current
   /* Search for the VPCA in the BTB list */
    while (current_block != NULL)
    {
        if (current_block->vpca == (VPCA & BTB_SIZE))
        {
          current_block->count = current_block->count + 1;
        }
        current_block = current_block->next_node;
    }
#endif
}

/* Handles inserting the targets into the BTB and replacement in case of the
 * BTB is full */
void vpc_predictor::insert_BTB(unsigned long int VPCA, unsigned long int CORRECT_TARGET)
{
#ifdef HASH
    //DEBUG// fprintf(stderr, "%d BTB pointer\n",current_block);
    VPCA = VPCA & BTB_SIZE;
    
    if(BTB == NULL)
    {
      BTB_struct* new_block = (BTB_struct*) malloc(sizeof(BTB_struct));
      new_block->count = 0;
      new_block->vpca = VPCA; 
      new_block->target = CORRECT_TARGET; 
      BTB = new_block;
      BTB->next_node = NULL; 
      return;
    }

    int block_count = 2;  // Initialize count
    int least_count = 1;  // Initialize count
    BTB_struct* last_block = BTB;
    BTB_struct* current_block = BTB;
    BTB_struct* least_count_entry_block = BTB;

    while (current_block != NULL)
    {
      if(current_block->vpca == VPCA)
      {
        current_block->target = CORRECT_TARGET; 
        current_block->count=1;
        return;
      }
      current_block = current_block->next_node;
    }

    while (last_block->next_node != NULL)
    {
        /* This variable now has number of blocks present in the set */
        block_count++;
        if(last_block->count <= least_count)
        {
          least_count_entry_block = last_block;
        }
        last_block = last_block->next_node;
    }

    /* block count is less than BTB_SIZE, adding a new node */
    if(block_count < BTB_SIZE)
    {
    BTB_struct* new_block = (BTB_struct*) malloc(sizeof(BTB_struct));
    new_block->count = 1;
    new_block->vpca = VPCA; 
    new_block->target = CORRECT_TARGET; 
    new_block->next_node = NULL; 
    last_block->next_node = new_block;
    }
    /* BTB is full, replacing the least count entry with the new VPCA and
     * target value */
    else
    {
      least_count_entry_block->count = 1;
      least_count_entry_block->vpca = VPCA; 
      least_count_entry_block->target = CORRECT_TARGET;
    }
#endif

#ifdef ARRAY
        targets[VPCA & BTB_SIZE] = CORRECT_TARGET; 
#endif
}

/* Function to predict target for a branch at PC */
unsigned long int vpc_predictor::predict(unsigned long int PC)
{
  /* Create a btb and assign the values in update stage */
  /* Need to access the BTB to get the target address */
  iter = 1;
  VPCA = PC;
  VGHR = hash_tron1.history_segment;
  unsigned long int VGHR_orig = hash_tron1.history_segment;
  done = false;

  while(!done)
  {
    pred_target = access_BTB(VPCA);
    pred_dir = hash_tron1.predict(VPCA);
    //fprintf(stderr, "sachin predict %llu in predict\n",VGHR);

    if(pred_target && (pred_dir == TAKEN))
    {
    /* Call conditional branch predictor with VPCA as address and  VGHR as
     * history_segment maybe?? */
      predicted_iter = iter;
      done = true;
      hash_tron1.history_segment = VGHR_orig;
      return pred_target;
    }
    else if(!pred_target || (iter >= MAX_ITER))
    {
      /* If not found assign gshares target maybe?? */
      done = true;
      hash_tron1.history_segment = VGHR_orig;
      return 0;
    }

  /* Update the VPCA with hash of pc, iter and VGHR with leftshift and
   * continue while with iter++ till done is true */
    VPCA = PC ^  VPC_CONSTANT[iter];
    VGHR = (VGHR << 1);
    VGHR &= 0xFFFFFFFFFFFFFFFF;
    hash_tron1.history_segment = VGHR;
    iter++;
  }

  hash_tron1.history_segment = VGHR_orig;
  return 0;
}

/* Function to train the predictor after the target for a branch at the PC is known */
void vpc_predictor::update_algo1(unsigned long int PC)
{
  /* Algorithm if the branch is correctly predicted */
  /* get the predicted iter from the above predict function */
  /* TODO: Condition this with correct prediction */
  iter = 1;
  VPCA = PC;

  /* While iter < predicted_iter do */
  /* If iter = predicted iter then update the BP as taken for that VPCA,
   * VGHR. Update the BTB with VPCA */
  while(iter <= predicted_iter)
  {
    if(iter == predicted_iter)
    {
      //update_conditional_BP(VPCA, VGHR,TAKEN);
      hash_tron1.update(VPCA, TAKEN);
      update_replacement_BTB(VPCA);
    }
  /* else update the BR as not taken for that VPCA and VGHR */ 
    else
    {
      hash_tron1.update(VPCA, NOT_TAKEN);
    }
  /* Repeat the loop with VPCA <- Hash(PC,iter) and VGHR <- lsl(BGHR) */
    /*assign VPCA <- PC, VGHR <- GHR */
    VPCA = PC ^  VPC_CONSTANT[iter];
    iter++;
  }
}

  /*TODO: Need to bypass this with a condition */
/* Function to train the predictor if the target is not found */
void vpc_predictor::update_algo2(unsigned long int PC, unsigned long int CORRECT_TARGET)
{

iter = 1;
VPCA = PC;
found_correct_target = false;

#ifdef HASH
  unsigned long int store_VPCA = PC;
#endif 

#ifdef ARRAY  
  unsigned long int store_VPCA = PC ^ VPC_CONSTANT[MAX_ITER];
  int least_out = 10;
#endif
  bool flag=0;

  while((iter <= MAX_ITER) && (found_correct_target == false))
  {
    pred_target = access_BTB(VPCA);
    if(pred_target == CORRECT_TARGET)
    {
      //update_conditional_BP(VPCA, VGHR,TAKEN);
      hash_tron1.update(VPCA, TAKEN);
#ifdef HASH
      update_replacement_BTB(VPCA);
#endif 
      found_correct_target = true;
    }
    else if(pred_target)
    {
      //update_conditional_BP(VPCA, NOT_TAKEN);
      hash_tron1.update(VPCA, NOT_TAKEN);
#ifdef ARRAY
      /* This is my own improvement to the existing algorithm, here I am
       * checking which VPCA has the least out value(ie highest chances of not
       * being taken */
     if((hash_tron1.out <= least_out) && !flag)
     store_VPCA = VPCA;
#endif
    }
    else if(!flag && !pred_target)
    {
      store_VPCA = VPCA;
      flag = 1;
    }

    /*assign VPCA <- PC, VGHR <- GHR */
    VPCA = PC ^  VPC_CONSTANT[iter];
    iter++;
  }

  if (found_correct_target == false) 
  {
  //DEBUG// fprintf(stderr, "sachin update 3 %d %ld in predict\n", VPCA,VGHR);
  insert_BTB(store_VPCA, CORRECT_TARGET);
  hash_tron1.update(store_VPCA, TAKEN);
  }
}


class my_update : public branch_update {
};

class my_predictor : public branch_predictor {
  public:
	branch_info bi;
  my_update u;

	my_predictor (void)  { 
	}

	branch_update *predict (branch_info & b) {
		bi = b;
		if (b.br_flags & BR_CONDITIONAL) 
    {
			u.direction_prediction (hash_tron.predict(b.address));
		}
    else 
    {
    u.direction_prediction (true);
		}

		if (b.br_flags & BR_INDIRECT) {
			u.target_prediction (vpc_pred.predict(b.address));
    }
		return &u;
	}

	void update (branch_update *u, bool taken, unsigned int target) {
		if (bi.br_flags & BR_CONDITIONAL) {
     hash_tron.update(bi.address, taken);
		}
		if (bi.br_flags & BR_INDIRECT) {
//DEBUG fprintf(stderr, "sachin address= %d, actual target = %d, predicted target = %d, actual direction = %d, predicted direction = %d, in predict\n", bi.address, target, vpc_pred.pred_target, taken, vpc_pred.pred_dir);
      if(taken && (vpc_pred.pred_target == target))
      {
        vpc_pred.update_algo1(bi.address);
      }
      else
      {
        vpc_pred.update_algo2(bi.address,target);
      }
    }
	}
};
