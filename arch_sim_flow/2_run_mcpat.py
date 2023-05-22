import os
class mcpat_runner:
    def __init__(self):
        self.num_of_config = 15
        self.benchmark=[
            "dhrystone",
            "median",
            "multiply",
            "qsort",
            "rsort",
            "spmv",
            "towers",
            "vvadd",
            "setup"
        ]
        self.clocks=[
            "500MHz",
            "1GHz",
            "2GHz"
        ]
    def run_mcpat(self):
        for i in range(self.num_of_config):
        #for i in range(1):
            for j in range(len(self.benchmark)):
                for k in range(len(self.clocks)):
                    workload = self.benchmark[j]
                    clk = self.clocks[k]
                    os.system("mkdir boom{}/{}/{}/mcpat_input".format(i,workload,clk))
                    os.system("python ../mcpat/mcpat-gem5/pars.py boom{}/{}/1GHz/stats.txt boom{}/{}/1GHz/config.json ../mcpat/mcpat-gem5/o3_{}.xml -o boom{}/{}/{}/mcpat_input".format(i,workload,i,workload,clk,i,workload,clk))
                    os.system("../mcpat/mcpat -infile /gem5/McPAT-Calib/boom{}/{}/{}/mcpat_input/mcpat-out-0.xml -opt_for_clk 1 -print_level 5 > /gem5/McPAT-Calib/boom{}/{}/{}/mcpat_result".format(i,workload,clk,i,workload,clk))

runner = mcpat_runner()
runner.run_mcpat()
