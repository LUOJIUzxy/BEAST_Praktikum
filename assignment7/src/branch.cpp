#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cassert>

// execute a list of commands, which modify values in memory
// - cmd 0: HALT - stop execution
// - cmd 1: NOP (no operation)
// - cmd 2: increment value at ctx[0]
// - cmd 3: negate value at ctx[0]
// - cmd 4: add ctx[1] to ctx[0]
// - cmd 5: save value at ctx[0] to ctx[1]
//
// different variants
// - variant A uses functions for command handlers
// - variant B uses a switch
// - variant C uses labels jumping to ("computed goto")

//---------------------------------------------------------------------
// variants A / B / C : use indirect calls/branches

// variant A

// signature for command handlers:
//   functions for different commands. return 1 to halt execution
typedef int (*runA_cmd_t)(int*);

// "(void) ctx;" prohibits compiler warning about unused paramter
int runA_cmd0(int* ctx) { (void) ctx;       return 1; } // HALT
int runA_cmd1(int* ctx) { (void) ctx;       return 0; } // NOP
int runA_cmd2(int* ctx) { ctx[0]++;         return 0; } // inc ctx[0]
int runA_cmd3(int* ctx) { ctx[0] = -ctx[0]; return 0; } // neg ctx[0]
int runA_cmd4(int* ctx) { ctx[0] += ctx[1]; return 0; } // add ctx[1]
int runA_cmd5(int* ctx) { ctx[1] = ctx[0];  return 0; } // save to ctx[1]

void runA(int size, char* cmds, int* ctx)
{
    runA_cmd_t cmdX[6] = { runA_cmd0, runA_cmd1, runA_cmd2,
                           runA_cmd3, runA_cmd4, runA_cmd5 };
    int i = 0;
    while(1) {
        runA_cmd_t cmd_func = cmdX[cmds[i]];
        if (cmd_func(ctx)) return;
        i++;
    }
}


// varinat B

void runB(int size, char* cmds, int* ctx)
{
    int i = 0;
    while(1) {
        switch(cmds[i]) {
            case 0: // HALT command: stop execution
                return;

            case 1: // NOP
                break;

            case 2: // increment value at ctx[0]
                ctx[0]++;
                break;

            case 3: // negate value at ctx[0]
                ctx[0] = -ctx[0];
                break;

            case 4: // add ctx[1] to ctx[0]
                ctx[0] += ctx[1];
                break;

            case 5: // save value at ctx[0] to ctx[1]
                ctx[1] = ctx[0];
                break;
        }
        i++;        
    }
}


// variant C

void runC(int size, char* cmds, int* ctx)
{
    // init with code addresses at labels, to jump via goto
    void* handler[6] = { &&cmd0, &&cmd1, &&cmd2, &&cmd3, &&cmd4, &&cmd5 };
    void* next;
    int i = 0;
    goto *handler[cmds[i++]];

cmd0: // HALT command: stop execution
    return;

cmd1:  // NOP
    goto *handler[cmds[i++]];

cmd2: // increment value at ctx[0]
    next = handler[cmds[i++]];
    ctx[0]++;
    goto *next;

cmd3:  // negate value at ctx[0]
    next = handler[cmds[i++]];
    ctx[0] = -ctx[0];
    goto *next;

cmd4:  // add ctx[1] to ctx[0]
    next = handler[cmds[i++]];
    ctx[0] += ctx[1];
    goto *next;

cmd5: // save value at ctx[0] to ctx[1]
    next = handler[cmds[i++]];
    ctx[1] = ctx[0];
    goto *next;
}


//---------------------------------------------------------------------
// variant D : use if-chains, ie conditional branches

void runD(int size, char* cmds, int* ctx)
{
    int i = 0;
    while(1) {
        int cmd = cmds[i];
        if (cmd > 3) {
            if (cmd == 4)
                ctx[0] += ctx[1]; // cmd 4: add ctx[1]
            else
                ctx[1] = ctx[0];  // cmd 5: save to ctx[1]
        }
        else if (cmd > 1) {
            if (cmd == 2)
                ctx[0]++;         // cmd 2: inc ctx[0]
            else
                ctx[0] = -ctx[0]; // cmd 3: neg ctx[0]
        }
        else  {
            if (cmd == 0)
                return;           // cmd 0: HALT
            else
                ;                 // cmd 1: NOP
        }
        i++;
    }
}

//---------------------------------------------------------------------
// variant E : indirect calls with recursion to test return prediction

typedef int (*runE_cmd_t)(int, char*, int*);

// forward decl
int runE_cmd0(int, char*, int*);
int runE_cmd1(int, char*, int*);
int runE_cmd2(int, char*, int*);
int runE_cmd3(int, char*, int*);
int runE_cmd4(int, char*, int*);
int runE_cmd5(int, char*, int*);

runE_cmd_t cmdE[6] = { runE_cmd0, runE_cmd1, runE_cmd2,
                       runE_cmd3, runE_cmd4, runE_cmd5 };

int runE_cmd0(int d, char* cmds, int* ctx)
{
    (void) d; (void) cmds; (void) ctx;
    // HALT
    return 10000;
}

int runE_cmd1(int d, char* cmds, int* ctx)
{
    // cmd 1: NOP

    if (d == 1) return 0;
    runE_cmd_t cmd_func = cmdE[*cmds];
    // +1 to force returns
    return cmd_func(d - 1, cmds + 1, ctx) + 1;
}

int runE_cmd2(int d, char* cmds, int* ctx)
{
    // cmd 2: inc ctx[0]
    ctx[0]++;

    if (d == 1) return 0;
    runE_cmd_t cmd_func = cmdE[*cmds];
    return cmd_func(d - 1, cmds + 1, ctx) + 1;
}

int runE_cmd3(int d, char* cmds, int* ctx)
{
    // cmd 3: neg ctx[0]
    ctx[0] = -ctx[0];

    if (d == 1) return 0;
    runE_cmd_t cmd_func = cmdE[*cmds];
    return cmd_func(d - 1, cmds + 1, ctx) + 1;
}

int runE_cmd4(int d, char* cmds, int* ctx)
{
    // cmd 4: add ctx[1]
    ctx[0] += ctx[1];

    if (d == 1) return 0;
    runE_cmd_t cmd_func = cmdE[*cmds];
    return cmd_func(d - 1, cmds + 1, ctx) + 1;
}

int runE_cmd5(int d, char* cmds, int* ctx)
{
    // cmd 5: save to ctx[1]
    ctx[1] = ctx[0];

    if (d == 1) return 0;
    runE_cmd_t cmd_func = cmdE[*cmds];
    return cmd_func(d - 1, cmds + 1, ctx) + 1;
}

// call chain depth
int E_depth = 1;

void runE(int size, char* cmds, int* ctx)
{
    while(1) {
        runE_cmd_t cmd_func = cmdE[*cmds];
        if (cmd_func(E_depth, cmds + 1, ctx) > 9999) return;
        cmds += E_depth;
    }
}


// signature of runX
typedef void (*run_t)(int, char*, int*);

void runReps(const char* name, int var,
             int reps, int size, char* cmds, int* ctx)
{
    // save start context values
    int ctx0 = ctx[0];
    int ctx1 = ctx[1];
    // int ctx0_final; // to assure we get same results

    if (var > 3) { E_depth = 1 + 5 * (var - 4); var = 4; }

    run_t runs[] = { runA, runB, runC, runD, runE };
    run_t run = runs[var];

    auto start = std::chrono::high_resolution_clock::now();
    for(int r = 0; r < reps; r++) {
        // restore start context values
        ctx[0] = ctx0;
        ctx[1] = ctx1;

        // execute command sequence
        (*run)(size, cmds, ctx);

        // check we get same ctx[0] value each time
        //if (r == 0) ctx0_final = ctx[0];
        //else if (ctx0_final != ctx[0])
        //    printf("Error: wrong result in rep %d\n", r);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(stop - start).count();

    if (name == 0) return; // warmup, do not print out statistics

    char estr[10] = { 0 };
    if (var == 4) snprintf(estr, 9, "%d", E_depth);
    printf("  %c%s-%s  %6d   %5.3fs   %5.3fns\n",
        'A' + var, estr, name, size, duration,
        1000000000.0 * duration / reps / size);

    // printf("  %c%s-%s  %6d   %5.3fs   %5.3fns (ctx0: %d)\n",
    //    'A' + var, estr, name, size, duration,
    //    1000000000.0 * duration / reps / size, ctx[0]);
}

// generate reproducible pseudo-random command sequence
void initRandom(int size, char* cmds, int seed)
{
    int val = seed; // PRNG start value

    for(int i = 0; i < size - 1; i++) {
        // simple PRNG: random 32 bit value sequence
        val = ((val * 1103515245U) + 12345U) & 0x7fffffff;

        // generate random commands 1 - 5
        cmds[i] = (val % 5) + 1;
    }
    // last command: HALT
    cmds[size-1] = 0;
}

// generate reproducible pseudo-random sequence with same command tuples
void initRandom2(int size, char* cmds, int seed)
{
    int val = seed; // PRNG start value

    for(int i = 0; i < size - 1; i+=2) {
        // simple PRNG: random 32 bit value sequence
        val = ((val * 1103515245U) + 12345U) & 0x7fffffff;

        // generate random commands 1 - 5
        cmds[i] = (val % 5) + 1;
        // every second command relates tp the previous
        cmds[i+1] = ((val+2) % 5) + 1;
    }
    // last command: HALT
    cmds[size-1] = 0;
}

// generate sequence consisting just of command <cmd>
void initFixed(int size, char* cmds, int cmd)
{
    for(int i = 0; i < size - 1; i++)
        cmds[i] = cmd;
    // last command: HALT
    cmds[size-1] = 0;
}

int main(int argc, char *argv[])
{
    // defaults
    // sequences up to size 500 k
    int maxsize = 500000;
    // run 300 mio commands for each measurement
    int total = 300000000;
    const char* vars = "ADEF";
    const char* seqs = "128";

    // arg parsing
    int arg = 1;
    while(arg < argc) {
        if (argv[arg][0] == '-') {
            if (argv[arg][1] == 'h') {
                printf("Usage: %s [-<vars>] [-<seqs>] [<max>]\n"
                       "\n"
                       " <vars>  variants to run (def: %s)\n"
                       " <seqs>  command sequences to run (def: %s)\n"
                       "         1-5: fixed / 8-9: random command streams\n"
                       " <max>   Run sequence sizes from 200 to <max> (def: %d)\n",
                        argv[0], vars, seqs, maxsize);
                exit(1);
            }
            if (argv[arg][1] >= 'A') vars = argv[arg]+1; // variants
            if (argv[arg][1] <= '9') seqs = argv[arg]+1; // sequences
        }
        else maxsize = atoi(argv[arg]);
        arg++;
    }
    if (maxsize == 0) maxsize = 500000;

    // print experiments which are executed
    fprintf(stderr, "Running variants \"%s\", sequences \"%s\",\n", vars, seqs);
    fprintf(stderr, "        max size %d, %d cmds per measurement\n", maxsize, total);
    fprintf(stderr, "Var-Seq  |  size |  total |  perCmd\n");
    fprintf(stderr, "---------+-------+--------+---------\n");

    // allocate buffer for command sequences
    char* cmds = new char[maxsize];
    // some arbitrary input values for command sequences
    int ctx[2] = {1, 2};

    // warm up
    initFixed(maxsize, cmds, 1);
    runReps(0, 0, total / maxsize, maxsize, cmds, ctx);

    // go over variants, run according code variant
    for(int i = 0; vars[i] != 0; i++) {
        int var = vars[i] - 'A';
        assert(var >= 0);

        // go over command sequence types
        for(int j = 0; seqs[j] != 0; j++) {
            int seq = seqs[j] - '0';
            // only accept 1-5 and 8/9
            char name[6]; // sequence name
            if ((seq > 0) && (seq < 6))
                snprintf(name, 6, "fix%1d", seq);
            else if ((seq == 8) || (seq == 9))
                snprintf(name, 6, "rnd%1d", seq - 7);
            else continue;

            for(int size = 200; size <= maxsize; size += size/2) {

                if (seq < 6)       initFixed(size, cmds, seq);
                else if (seq == 8) initRandom(size, cmds, 1);
                else if (seq == 9) initRandom2(size, cmds, 1);

                runReps(name, var, total / size, size, cmds, ctx);
            }
        }
    }
}

