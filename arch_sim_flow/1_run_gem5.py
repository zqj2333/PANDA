import os
class gem5_runner:
    def __init__(self):
        self.boom_core_config_table = [
            #0          1           2                3        4               5              6            7           8                     9             10               11                    12         13
            #FetchWidth DecodeWidth FetchBufferEntry RobEntry IntPhysRegister FpPhysRegister LDQ/STQEntry BranchCount MemIssue/FpIssueWidth IntIssueWidth DCache/ICacheWay DCacheTLBEntry        DCacheMSHR ICacheFetchBytes
            [4,         1,          5,               16,      36,             36,            4,           6,          1,                    1,            2,               8,                    2,         2],
            [4,         1,          8,               32,      53,             48,            8,           8,          1,                    1,            4,               8,                    2,         2],
            [4,         1,          16,              48,      68,             56,            16,          10,         1,                    1,            8,               16,                   4,         2],

            [4,         2,          8,               64,      64,             56,            12,          10,         1,                    1,            4,               8,                    2,         2],
            [4,         2,          16,              64,      80,             64,            16,          12,         1,                    2,            4,               8,                    2,         2],
            [8,         2,          24,              80,      88,             72,            20,          14,         1,                    2,            8,               16,                   4,         4],

            [8,         3,          18,              81,      88,             88,            16,          14,         1,                    2,            8,               16,                   4,         4],
            [8,         3,          24,              96,      110,            96,            24,          16,         1,                    3,            8,               16,                   4,         4],
            [8,         3,          30,              114,     112,            112,           32,          16,         2,                    3,            8,               32,                   4,         4],

            [8,         4,          24,              112,     108,            108,           24,          18,         1,                    4,            8,               32,                   4,         4],
            [8,         4,          32,              128,     128,            128,           32,          20,         2,                    4,            8,               32,                   4,         4],
            [8,         4,          40,              136,     136,            136,           36,          20,         2,                    4,            8,               32,                   8,         4],

            [8,         5,          30,              125,     108,            108,           24,          18,         2,                    5,            8,               32,                   8,         4],
            [8,         5,          35,              130,     128,            128,           32,          20,         2,                    5,            8,               32,                   8,         4],
            [8,         5,          40,              140,     140,            140,           36,          20,         2,                    5,            8,               32,                   8,         4]
        ]
        self.benchmark=[
            "setup",
            "dhrystone"
            "median",
            "multiply",
            "qsort",
            "rsort",
            "spmv",
            "towers",
            "vvadd"
        ]
        self.clocks=[
            "500MHz",
            "1GHz",
            "2GHz"
        ]

    def run_gem5(self):
        for i in range(len(self.boom_core_config_table)):
            config = self.boom_core_config_table[i]
            isu_params = [
            # IQT_MEM.numEntries IQT_MEM.dispatchWidth
            # IQT_INT.numEntries IQT_INT.dispatchWidth
            # IQT_FP.numEntries IQT_FP.dispatchWidth
            [8, config[1], 8, config[1], 8, config[1]],
            [12, config[1], 20, config[1], 16, config[1]],
            [16, config[1], 32, config[1], 24, config[1]],
            [24, config[1], 40, config[1], 32, config[1]],
            [24, config[1], 40, config[1], 32, config[1]]
            ]
            _isu_params = isu_params[config[1] - 1]
            #gem5 config
            dcachesize = config[10]*64*64/1024
            dcacheassoc = config[10]
            dcachemshrs = config[12]
            icachesize = config[10]*64*64/1024
            icacheassoc = config[10]
            fetchWidth = config[0]
            decodeWidth = config[1]
            numROBEntries = config[3]
            LQEntries = config[6]
            SQEntries = config[6]
            numPhysIntRegs = config[4]
            numPhysFloatRegs = config[5]
            if config[2]>=16:
                fetchBufferSize = 64
            elif config[2]>=8:
                fetchBufferSize = 32
            elif config[2]>=4:
                fetchBufferSize = 16
            else:
                fetchBufferSize = 8
            commitWidth = decodeWidth
            renameWidth = decodeWidth
            issueWidth = config[8]*2+config[9]
            dispatchWidth = config[1]*3
            if dispatchWidth>12:
                dispatchWidth=12
            wbWidth = issueWidth
            squashWidth = issueWidth
            numIQEntries = _isu_params[0]+_isu_params[2]+_isu_params[4]
            itbsize = config[11]
            dtbsize = config[11]
            fuPool = "Boom_{}_{}_FUPool()".format(config[8],config[9])
            #end
            config_file='''
import argparse
import sys
import os

import m5
from m5.defines import buildEnv
from m5.objects import *
from m5.params import NULL
from m5.util import addToPath, fatal, warn

addToPath('../')

from ruby import Ruby

from common import Options
from common import Simulation
from common import CacheConfig
from common import CpuConfig
from common import ObjectList
from common import MemConfig
from common.FileSystemConfig import config_filesystem
from common.Caches import *
from common.cpu2000 import *

def get_processes(args):
    """Interprets provided args and returns a list of processes"""

    multiprocesses = []
    inputs = []
    outputs = []
    errouts = []
    pargs = []

    workloads = args.cmd.split(';')
    if args.input != "":
        inputs = args.input.split(';')
    if args.output != "":
        outputs = args.output.split(';')
    if args.errout != "":
        errouts = args.errout.split(';')
    if args.options != "":
        pargs = args.options.split(';')

    idx = 0
    for wrkld in workloads:
        process = Process(pid = 100 + idx)
        process.executable = wrkld
        process.cwd = os.getcwd()
        process.gid = os.getgid()

        if args.env:
            with open(args.env, 'r') as f:
                process.env = [line.rstrip() for line in f]

        if len(pargs) > idx:
            process.cmd = [wrkld] + pargs[idx].split()
        else:
            process.cmd = [wrkld]

        if len(inputs) > idx:
            process.input = inputs[idx]
        if len(outputs) > idx:
            process.output = outputs[idx]
        if len(errouts) > idx:
            process.errout = errouts[idx]

        multiprocesses.append(process)
        idx += 1

    if args.smt:
        assert(args.cpu_type == "DerivO3CPU")
        return multiprocesses, idx
    else:
        return multiprocesses, 1


parser = argparse.ArgumentParser()
Options.addCommonOptions(parser)
Options.addSEOptions(parser)

if '--ruby' in sys.argv:
    Ruby.define_options(parser)

args = parser.parse_args()

multiprocesses = []
numThreads = 1

if args.bench:
    apps = args.bench.split("-")
    if len(apps) != args.num_cpus:
        print("number of benchmarks not equal to set num_cpus!")
        sys.exit(1)

    for app in apps:
        try:
            if buildEnv['TARGET_ISA'] == 'arm':
                exec("workload = %%s('arm_%%s', 'linux', '%%s')" %% (
                        app, args.arm_iset, args.spec_input))
            else:
                exec("workload = %%s(buildEnv['TARGET_ISA', 'linux', '%%s')" %% (
                        app, args.spec_input))
            multiprocesses.append(workload.makeProcess())
        except:
            print("Unable to find workload for %%s: %%s" %%
                  (buildEnv['TARGET_ISA'], app),
                  file=sys.stderr)
            sys.exit(1)
elif args.cmd:
    multiprocesses, numThreads = get_processes(args)
else:
    print("No workload specified. Exiting!\\n", file=sys.stderr)
    sys.exit(1)


(CPUClass, test_mem_mode, FutureClass) = Simulation.setCPUClass(args)
CPUClass.numThreads = numThreads

# Check -- do not allow SMT with multiple CPUs
if args.smt and args.num_cpus > 1:
    fatal("You cannot use SMT with multiple CPUs!")

np = args.num_cpus
mp0_path = multiprocesses[0].executable
system = System(cpu = [CPUClass(cpu_id=i) for i in range(np)],
                mem_mode = test_mem_mode,
                mem_ranges = [AddrRange(args.mem_size)],
                cache_line_size = args.cacheline_size)

if numThreads > 1:
    system.multi_thread = True

# Create a top-level voltage domain
system.voltage_domain = VoltageDomain(voltage = args.sys_voltage)

# Create a source clock for the system and set the clock period
system.clk_domain = SrcClockDomain(clock =  args.sys_clock,
                                   voltage_domain = system.voltage_domain)

# Create a CPU voltage domain
system.cpu_voltage_domain = VoltageDomain()

# Create a separate clock domain for the CPUs
system.cpu_clk_domain = SrcClockDomain(clock = args.cpu_clock,
                                       voltage_domain =
                                       system.cpu_voltage_domain)

# If elastic tracing is enabled, then configure the cpu and attach the elastic
# trace probe
if args.elastic_trace_en:
    CpuConfig.config_etrace(CPUClass, system.cpu, args)

# All cpus belong to a common cpu_clk_domain, therefore running at a common
# frequency.
for cpu in system.cpu:
    cpu.clk_domain = system.cpu_clk_domain

if ObjectList.is_kvm_cpu(CPUClass) or ObjectList.is_kvm_cpu(FutureClass):
    if buildEnv['TARGET_ISA'] == 'x86':
        system.kvm_vm = KvmVM()
        system.m5ops_base = 0xffff0000
        for process in multiprocesses:
            process.useArchPT = True
            process.kvmInSE = True
    else:
        fatal("KvmCPU can only be used in SE mode with x86")

# Sanity check
if args.simpoint_profile:
    if not ObjectList.is_noncaching_cpu(CPUClass):
        fatal("SimPoint/BPProbe should be done with an atomic cpu")
    if np > 1:
        fatal("SimPoint generation not supported with more than one CPUs")

for i in range(np):
    if args.smt:
        system.cpu[i].workload = multiprocesses
    elif len(multiprocesses) == 1:
        system.cpu[i].workload = multiprocesses[0]
    else:
        system.cpu[i].workload = multiprocesses[i]

    if args.simpoint_profile:
        system.cpu[i].addSimPointProbe(args.simpoint_interval)

    if args.checker:
        system.cpu[i].addCheckerCpu()

    if args.bp_type:
        bpClass = ObjectList.bp_list.get(args.bp_type)
        system.cpu[i].branchPred = bpClass()

    if args.indirect_bp_type:
        indirectBPClass = \\
            ObjectList.indirect_bp_list.get(args.indirect_bp_type)
        system.cpu[i].branchPred.indirectBranchPred = indirectBPClass()

    system.cpu[i].createThreads()

if args.ruby:
    Ruby.create_system(args, False, system)
    assert(args.num_cpus == len(system.ruby._cpu_ports))

    system.ruby.clk_domain = SrcClockDomain(clock = args.ruby_clock,
                                        voltage_domain = system.voltage_domain)
    for i in range(np):
        ruby_port = system.ruby._cpu_ports[i]

        # Create the interrupt controller and connect its ports to Ruby
        # Note that the interrupt controller is always present but only
        # in x86 does it have message ports that need to be connected
        system.cpu[i].createInterruptController()

        # Connect the cpu's cache ports to Ruby
        ruby_port.connectCpuPorts(system.cpu[i])
else:
    MemClass = Simulation.setMemClass(args)
    system.membus = SystemXBar()
    system.system_port = system.membus.cpu_side_ports
    CacheConfig.config_cache(args, system)
    MemConfig.config_mem(args, system)
    config_filesystem(system, args)

#my own config

system.cpu[0].dcache.size="%dkB"
system.cpu[0].dcache.assoc=%d
system.cpu[0].dcache.mshrs=%d
system.cpu[0].dcache.replacement_policy=RandomRP()
system.cpu[0].icache.size="%dkB"
system.cpu[0].icache.assoc=%d#ICache nWays
system.cpu[0].icache.replacement_policy=RandomRP()
#gem5/src/cpu/o3/BaseO3CPU.py
system.cpu[0].fetchWidth=%d#FetchWidth
system.cpu[0].decodeWidth=%d#DecodeWidth,corewidth,equal to decode width, integer rename width, ROB width, commit width
system.cpu[0].numROBEntries=%d#RobEntry
system.cpu[0].LQEntries=%d#LDQEntry
system.cpu[0].SQEntries=%d#STQEntry
system.cpu[0].numPhysIntRegs=%d#IntPhysRegister
system.cpu[0].numPhysFloatRegs=%d#FpPhysRegister
system.cpu[0].fetchBufferSize=%d#FetchBufferEntry
system.cpu[0].commitWidth=%d#DecodeWidth
system.cpu[0].renameWidth=%d#DecodeWidth
system.cpu[0].issueWidth=%d#MemIssue+FpIssue+IntIssue
system.cpu[0].dispatchWidth=%d#MemDispatch+FpDispatch+IntDispatch
system.cpu[0].wbWidth=%d#IssueUnit
system.cpu[0].squashWidth=%d#IssueUnit
system.cpu[0].numIQEntries=%d#16*IssueUnit
system.cpu[0].mmu.itb.size=%d#DCache/ICacheTLBEntry
system.cpu[0].mmu.dtb.size=%d#DCache/ICacheTLBEntry
#gem5/src/cpu/o3/FuncUnitConfig.py
system.cpu[0].fuPool=%s#Equal to IssueUnit


#end

system.workload = SEWorkload.init_compatible(mp0_path)

if args.wait_gdb:
    system.workload.wait_for_remote_gdb = True

root = Root(full_system = False, system = system)
Simulation.run(args, root, system, FutureClass)
''' % (
    dcachesize,
    dcacheassoc,
    dcachemshrs,
    icachesize,
    icacheassoc,
    fetchWidth,
    decodeWidth,
    numROBEntries,
    LQEntries,
    SQEntries,
    numPhysIntRegs,
    numPhysFloatRegs,
    fetchBufferSize,
    commitWidth,
    renameWidth,
    issueWidth,
    dispatchWidth,
    wbWidth,
    squashWidth,
    numIQEntries,
    itbsize,
    dtbsize,
    fuPool
)
            os.system("touch /gem5/gem5/configs/example/boom_config.py")
            with open("/gem5/gem5/configs/example/boom_config.py", 'w') as f:
                f.writelines(config_file)
            os.system("mkdir boom{}".format(i))
            for j in range(len(self.benchmark)):
                workload = self.benchmark[j]
                os.system("mkdir boom{}/{}".format(i,workload))
                for k in range(len(self.clocks)):
                    clk = self.clocks[k]
                    os.system("mkdir boom{}/{}/{}".format(i,workload,clk))
                    script = "cd /gem5/gem5\nbuild/RISCV/gem5.opt configs/example/boom_config.py --sys-clock={} --cpu-clock={} --cmd=mytest/{}/{}.riscv  --caches --l1i_size=16kB --l1i_assoc=4 --l1d_assoc=4 --l1d_size=16kB --cacheline=64 --mem-size=512MB --cpu-type=RiscvO3CPU > /gem5/McPAT-Calib/boom{}/{}/{}/gem5log 2>&1".format(clk,clk,workload,workload,i,workload,clk)
                    os.system("touch script.sh")
                    with open("script.sh", 'w') as f:
                        f.writelines(script)
                    os.system("bash script.sh")
                    os.system("rm script.sh")
                    os.system("cp /gem5/gem5/m5out/config* boom{}/{}/{}".format(i,workload,clk))
                    os.system("cp /gem5/gem5/m5out/stats.txt boom{}/{}/{}".format(i,workload,clk))
                #break
            os.system("rm /gem5/gem5/configs/example/boom_config.py")
            #break

runner = gem5_runner()
runner.run_gem5()
            

        
