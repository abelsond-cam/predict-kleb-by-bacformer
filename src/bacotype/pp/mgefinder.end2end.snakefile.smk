# The Snakefile for MGEfinder which runs bwa, and indexes the bam file and  the assembly before running the standard MGEfinder pipeline. It also removes the bam file after finishing. This is the command I used to run it at the Sanger.
# snakemake -s mgefinder.end2end.snakefile --configfile /nfs/users/nfs_a/aw27/aw27/conda/mgefinder/lib/python3.8/site-packages/mgefinder-1.0.6-py3.8.egg/mgefinder/workflow/denovo.original.config.yml   --config memory=16000 wd=  --profile /nfs/users/nfs_a/aw27/.config/snakemake/lsf --default-resources mem_mb=16000 --restart-times 3 --rerun-incomplete
# It currently expects a table mgefinder_dataset.txt with the file locations which is parsed by Snakemake (at the begining of the pipeline) into dictionaries. If all the assemblies/reads are in the the same directory, this can be simplified. You'll have to re-download the reads yourself. Probably easier to explain on a quick call.

conda: "dependencies.yaml"

import os
from os.path import basename, join

WD = "" 

GENOME_DIR= join(WD, config['genome_dir'])
ASSEMBLY_DIR = join(WD, config['assembly_dir'])
BAM_DIR = join(WD, config['bam_dir'])
MUSTACHE_DIR = join(WD, config['mgefinder_dir'])
DATABASE_DIR = join(WD, config['database_dir'])
RESULTS_DIR = join(WD, config['results_dir'])

WC_genomes = glob_wildcards(join(GENOME_DIR, '{genome}.fna'))
WC_bam_samples = glob_wildcards(join(BAM_DIR, "{sample}.{genome}.bam"))

meta = "mgefinder_dataset.txt"

def get_assembly_dict():
    sample2filename = {}
    with open(meta) as f:
        f.readline() 
        for l in f:
            data_dir, sample_id, sample_name, gff, contigs = l.strip().split('\t') 
            sample2filename[sample_name] = contigs
    return sample2filename 

def get_data_dict():
    sample2filename = {}
    with open(meta) as f:
        f.readline() 
        for l in f:
            data_dir, sample_id, sample_name, gff, contigs = l.strip().split('\t') 
            sample2filename[sample_name] = data_dir 
    return sample2filename 

def get_sample_dict():
    sample2filename = {}
    with open(meta) as f:
        f.readline() 
        for l in f:
            data_dir, sample_id, sample_name, gff, contigs = l.strip().split('\t') 
            sample2filename[sample_name] = sample_id 
    return sample2filename 

def get_assembly():
    sample2filename = {}
    with open(meta) as f:
        f.readline() 
        for l in f:
            data_dir, sample_id, sample_name, gff, contigs = l.strip().split('\t') 
            sample2filename[sample_name] = contigs
    return sample2filename 

assembly_dict = get_assembly_dict()
data_dict = get_data_dict()
sample_dict = get_sample_dict()
    
def get_samples():
    samples = [] 
    with open(meta) as f:
        f.readline() 
        i = 0
        for l in f:
            #if i == 1:
            #    break
            i+=1
            data_dir, sample_id, sample_name, gff, contigs = l.strip().split('\t') 
            samples.append(sample_name)
    return samples 


SAMPLES = get_samples() 
GENOMES = ['PAO1']


rule all:
    input:
        expand(join(RESULTS_DIR, '{genome}/04.makefasta.{genome}.all_seqs.fna'), genome=GENOMES)
    run:
        pass

rule copy_genome:
    input:
        "/lustre/scratch125/pam/teams/team216/aw27/all_paerug/transposon_insertions/mgefinder/bwa/{genome}.fna"
    output:
        "00.genome/{genome}.fna"
    shell:
        """cp {input} {output}"""

rule copy_assembly:
    input:
        assembly = lambda wc: assembly_dict[wc.sample],  
    output:
        "00.assembly/{sample}.fna"
    shell:
        """ln -s {input} {output} """

rule formatbam:
    input:
        "bwa/{sample}.{genome}.bwa.sam"
    output:
        temp("00.bam/{sample}.{genome}.bam")
    shell:
        """mgefinder formatbam  {input} {output}"""


rule bwa_index:
    input: 
        "00.genome/{genome}.fna"
    output:
        "00.genome/{genome}.fna.amb",
    shell:
        """bwa index {input} {output}"""

rule bwa:
    input:
        index = "00.genome/{genome}.fna.amb",
        genome = "00.genome/{genome}.fna", 
        fastq1 = lambda wc: data_dict[wc.sample] + "/" + sample_dict[wc.sample] + "_1.fastq.gz",  
        fastq2 = lambda wc: data_dict[wc.sample] + "/" + sample_dict[wc.sample] + "_2.fastq.gz"  

    params:
        #data_dir = lambda wc: data_dict[wc.sample],       
        #sample_id = lambda wc: sample_dict[wc.sample],
    output: 
        sample=temp("bwa/{sample}.{genome}.bwa.sam")
    shell:
        """bwa mem {input.genome} {input.fastq1} {input.fastq2}   -o  {output.sample}"""
        

rule index_genome_bowtie2:
    log: join(GENOME_DIR, "log/{genome}.index_bowtie2.log")
    benchmark: join(GENOME_DIR, "log/{genome}.index_bowtie2.benchmark.txt")
    input:
        join(GENOME_DIR, "{genome}.fna")
    output:
        one=join(GENOME_DIR, "{genome}.fna.1.bt2"),
        two=join(GENOME_DIR, "{genome}.fna.2.bt2"),
        three=join(GENOME_DIR, "{genome}.fna.3.bt2"),
        four=join(GENOME_DIR, "{genome}.fna.4.bt2"),
        revone=join(GENOME_DIR, "{genome}.fna.rev.1.bt2"),
        revtwo=join(GENOME_DIR, "{genome}.fna.rev.2.bt2")
    shell:
        """
        bowtie2-build {input} {input} 1> {log} 2> {log}.err || \
        (cat {log}.err; exit 1)
        """

rule index_assembly:
    log: join(ASSEMBLY_DIR, "log/{sample}.index_assembly.log")
    benchmark: join(ASSEMBLY_DIR, "log/{sample}.index_assembly.benchmark.txt")
    input:
        contigs=join(ASSEMBLY_DIR, "{sample}.fna")
    output:
        one=join(ASSEMBLY_DIR, "{sample}.fna.1.bt2"),
        two=join(ASSEMBLY_DIR, "{sample}.fna.2.bt2"),
        three=join(ASSEMBLY_DIR, "{sample}.fna.3.bt2"),
        four=join(ASSEMBLY_DIR, "{sample}.fna.4.bt2"),
        revone=join(ASSEMBLY_DIR, "{sample}.fna.rev.1.bt2"),
        revtwo=join(ASSEMBLY_DIR, "{sample}.fna.rev.2.bt2")
    shell:
        """
        bowtie2-build {input} {input} 1> {log} 2> {log}.err || \
        (cat {log}.err; exit 1)
        """


rule find:
    log: join(MUSTACHE_DIR, "{genome}/{sample}/log/{sample}.{genome}.find.log")
    benchmark: join(MUSTACHE_DIR, "{genome}/{sample}/log/{sample}.{genome}.find.benchmark.txt")
    input:
        bam=join(BAM_DIR, "{sample}.{genome}.bam"),
    output:
        find=join(MUSTACHE_DIR, '{genome}/{sample}/01.find.{sample}.{genome}.tsv')
    params:
        sample='{sample}',
        minlen=config['find']['minlen'],
        mincount=config['find']['mincount'],
        minq=config['find']['minq'],
        minial=config['find']['minial'],
        mindist=config['find']['mindist'],
        minratio=config['find']['minratio'],
        maxir=config['find']['maxir'],
        lins=config['find']['lins'],
        mcc=config['find']['mcc'],
        check_bwa=config['find']['check_bwa_flag']
    shell:
        """
        mgefinder find -id {params.sample} -minlen {params.minlen} -mincount {params.mincount} -minq {params.minq} \
        -minial {params.minial} -mindist {params.mindist} -minratio {params.minratio} -maxir {params.maxir} -lins \
        {params.lins} -mcc {params.mcc} {params.check_bwa} {input.bam} -o {output.find} 1> {log} 2> {log}.err || \
        (cat {log}.err; exit 1)
        """


rule pair:
    log: join(MUSTACHE_DIR, "{genome}/{sample}/log/{sample}.{genome}.pair.log")
    benchmark: join(MUSTACHE_DIR, "{genome}/{sample}/log/{sample}.{genome}.pair.benchmark.txt")
    input:
        find=ancient(join(MUSTACHE_DIR, '{genome}/{sample}/01.find.{sample}.{genome}.tsv')),
        bam=join(BAM_DIR, "{sample}.{genome}.bam"),
        genome=join(GENOME_DIR, "{genome}.fna")
    output:
        pair=join(MUSTACHE_DIR, '{genome}/{sample}/02.pair.{sample}.{genome}.tsv')
    params:
        maxdr=config['pair']['maxdr'],
        minq=config['pair']['minq'],
        minial=config['pair']['minial'],
        maxjsp=config['pair']['maxjsp'],
        lins=config['pair']['lins']
    shell:
        """
        mgefinder pair -maxdr {params.maxdr} -minq {params.minq} -minial {params.minial} -maxjsp {params.maxjsp} \
        -lins {params.lins} {input.find} {input.bam} {input.genome} -o {output.pair} 1> {log} 2> {log}.err || \
        (cat {log}.err; exit 1)
        """


rule inferseq_assembly:
    log: join(MUSTACHE_DIR, "{genome}/{sample}/log/{sample}.{genome}.inferseq_assembly.log")
    benchmark: join(MUSTACHE_DIR, "{genome}/{sample}/log/{sample}.{genome}.inferseq_assembly.benchmark.txt")
    input:
        pair=join(MUSTACHE_DIR, '{genome}/{sample}/02.pair.{sample}.{genome}.tsv'),
        bam=join(BAM_DIR, "{sample}.{genome}.bam"),
        recover_reference=join(GENOME_DIR, "{genome}.fna"),
        recover_assembly=join(ASSEMBLY_DIR, "{sample}.fna"),
        one_ref=join(GENOME_DIR, '{genome}.fna.1.bt2'),
        one_asm=join(ASSEMBLY_DIR, '{sample}.fna.1.bt2')
    output:
        recover=join(MUSTACHE_DIR, '{genome}/{sample}/03.inferseq_assembly.{sample}.{genome}.tsv')
    params:
        minident=config['inferseq_assembly']['minident'],
        maxclip=config['inferseq_assembly']['maxclip'],
        maxsize=config['inferseq_assembly']['maxsize'],
        minsize=config['inferseq_assembly']['minsize']
    shell:
        """
        mgefinder inferseq-assembly -minident {params.minident} -maxclip {params.maxclip} -maxsize {params.maxsize} \
        -minsize {params.minsize} {input.pair} {input.bam} {input.recover_assembly} {input.recover_reference} -o {output.recover} 1> {log} 2> {log}.err || \
        (cat {log}.err; exit 1)
        """


rule inferseq_reference:
    log: join(MUSTACHE_DIR, "{genome}/{sample}/log/{sample}.{genome}.inferseq_reference.log")
    benchmark: join(MUSTACHE_DIR, "{genome}/{sample}/log/{sample}.{genome}.inferseq_reference.benchmark.txt")
    input:
        pair=join(MUSTACHE_DIR, '{genome}/{sample}/02.pair.{sample}.{genome}.tsv'),
        recover_reference=join(GENOME_DIR, "{genome}.fna"),
        one_ref=join(GENOME_DIR, '{genome}.fna.1.bt2')
    output:
        recover=join(MUSTACHE_DIR, '{genome}/{sample}/03.inferseq_reference.{sample}.{genome}.tsv')
    params:
        minident=config['inferseq_reference']['minident'],
        maxclip=config['inferseq_reference']['maxclip'],
        maxsize=config['inferseq_reference']['maxsize'],
        minsize=config['inferseq_reference']['minsize']
    shell:
        """
        mgefinder inferseq-reference -minident {params.minident} -maxclip {params.maxclip} -maxsize {params.maxsize} \
        -minsize {params.minsize} {input.pair} {input.recover_reference} -o {output.recover} 1> {log} 2> {log}.err || \
        (cat {log}.err; exit 1)
        """


rule inferseq_overlap:
    log: join(MUSTACHE_DIR, "{genome}/{sample}/log/{sample}.{genome}.inferseq_overlap.log")
    benchmark: join(MUSTACHE_DIR, "{genome}/{sample}/log/{sample}.{genome}.benchmark.txt")
    input:
        pair=join(MUSTACHE_DIR, '{genome}/{sample}/02.pair.{sample}.{genome}.tsv'),
    output:
        outfile=join(MUSTACHE_DIR, '{genome}/{sample}/03.inferseq_overlap.{sample}.{genome}.tsv')
    params:
        minscore=config['inferseq_overlap']['minscore'],
        minopi=config['inferseq_overlap']['minopi'],
        minsize=config['inferseq_overlap']['minsize']
    shell:
        """
        mgefinder inferseq-overlap -minscore {params.minscore} -minopi {params.minopi} -minsize {params.minsize} \
        {input.pair} -o {output.outfile} 1> {log} 2> {log}.err
        """

rule make_inferseq_file_path_list:
    input:
        expand(join(MUSTACHE_DIR, '{{genome}}/{sample}/03.inferseq_assembly.{sample}.{{genome}}.tsv'), sample=SAMPLES),
        expand(join(MUSTACHE_DIR, '{{genome}}/{sample}/03.inferseq_reference.{sample}.{{genome}}.tsv'), sample=SAMPLES),
        expand(join(MUSTACHE_DIR, '{{genome}}/{sample}/03.inferseq_overlap.{sample}.{{genome}}.tsv'), sample=SAMPLES)
    output:
        outfile=join(MUSTACHE_DIR, '{genome}/{genome}.all_inferseq.txt')
    run:
        with open(output.outfile, 'w') as outfile:
            for f in input:
                outfile.write(f+'\n')

rule make_database:
    benchmark: join(DATABASE_DIR, '{genome}/{genome}.database.benchmark.txt')
    input:
        join(MUSTACHE_DIR, '{genome}/{genome}.all_inferseq.txt')
    output:
        db=join(DATABASE_DIR, '{genome}/{genome}.database.fna'),
        index=join(DATABASE_DIR, '{genome}/{genome}.database.fna.1.bt2')
    threads:
        16
    params:
        memory=config['memory'],
        outdir=join(DATABASE_DIR, '{genome}'),
        prefix='{genome}.database',
        minsize=config['makedatabase']['minsize'],
        maxsize=config['makedatabase']['maxsize'],
    shell:
        """
        mgefinder makedatabase -minsize {params.minsize} -maxsize {params.maxsize} --threads {threads} \
        --memory {params.memory} -o {params.outdir} -p {params.prefix} {input}  --force
        """


rule inferseq_database:
    log: join(MUSTACHE_DIR, "{genome}/{sample}/log/{sample}.{genome}.inferseq_database.log")
    benchmark: join(MUSTACHE_DIR, "{genome}/{sample}/log/{sample}.{genome}.inferseq_database.benchmark.txt")
    input:
        pair=join(MUSTACHE_DIR, '{genome}/{sample}/02.pair.{sample}.{genome}.tsv'),
        database=join(DATABASE_DIR, '{genome}/{genome}.database.fna'),
        one=join(DATABASE_DIR, '{genome}/{genome}.database.fna.1.bt2')
    output:
        outfile=join(MUSTACHE_DIR, '{genome}/{sample}/04.inferseq_database.{sample}.{genome}.tsv')
    params:
        minident=config['inferseq_database']['minident'],
        maxclip=config['inferseq_database']['maxclip'],
        maxedgedist=config['inferseq_database']['maxedgedist']
    shell:
        """
        mgefinder inferseq-database -minident {params.minident} -maxclip {params.maxclip} -maxedgedist \
        {params.maxedgedist} {input.pair} {input.database} -o {output.outfile} 1> {log} 2> {log}.err
        """

rule make_inferseq_database_file_path_list:
    input:
        expand(join(MUSTACHE_DIR, '{{genome}}/{sample}/03.inferseq_assembly.{sample}.{{genome}}.tsv'), sample=SAMPLES),
        expand(join(MUSTACHE_DIR, '{{genome}}/{sample}/03.inferseq_reference.{sample}.{{genome}}.tsv'), sample=SAMPLES),
        expand(join(MUSTACHE_DIR, '{{genome}}/{sample}/03.inferseq_overlap.{sample}.{{genome}}.tsv'), sample=SAMPLES),
        expand(join(MUSTACHE_DIR, '{{genome}}/{sample}/04.inferseq_database.{sample}.{{genome}}.tsv'), sample=SAMPLES)
    output:
        outfile=join(MUSTACHE_DIR, '{genome}/{genome}.all_inferseq_database.txt')
    run:
        with open(output.outfile, 'w') as outfile:
            for f in input:
                outfile.write(f+'\n')

rule clusterseq:
    input:
        join(MUSTACHE_DIR, '{genome}/{genome}.all_inferseq_database.txt')
    output:
        clusterseq=join(RESULTS_DIR, '{genome}/01.clusterseq.{genome}.tsv')
    threads:
        16
    params:
        memory=config['memory'],
        prefix='{genome}.database',
        minsize=config['clusterseq']['minsize'],
        maxsize=config['clusterseq']['maxsize'],
    shell:
        """
        mgefinder clusterseq -minsize {params.minsize} -maxsize {params.maxsize} --threads {threads} \
        --memory {params.memory} {input} -o {output}
        """

rule make_pair_file_path_list:
    input:
        expand(join(MUSTACHE_DIR, '{{genome}}/{sample}/02.pair.{sample}.{{genome}}.tsv'), sample=SAMPLES)
    output:
        outfile=join(MUSTACHE_DIR, '{genome}/{genome}.all_pair.txt')
    run:
        with open(output.outfile, 'w') as outfile:
            for f in input:
                outfile.write(f+'\n')


rule genotype:
    log: join(RESULTS_DIR, "{genome}/log/{genome}.genotype.log")
    benchmark: join(RESULTS_DIR, "{genome}/log/{genome}.genotype.benchmark.txt")
    input:
        join(RESULTS_DIR, '{genome}/01.clusterseq.{genome}.tsv'),
        outfile=join(MUSTACHE_DIR, '{genome}/{genome}.all_pair.txt')
    output:
        genotype=join(RESULTS_DIR, '{genome}/02.genotype.{genome}.tsv')
    params:
        filter_clusters=config['genotype']['filter_clusters']
    shell:
        """
        if [ "{params.filter_clusters}" == "True" ]; then
            mgefinder genotype --filter-clusters-inferred-assembly {input} -o {output} 1> {log} 2> {log}.err || \
            (cat {log}.err; exit 1)
        else
            mgefinder genotype --no-filter-clusters-inferred-assembly {input} -o {output} 1> {log} 2> {log}.err || \
            (cat {log}.err; exit 1)
        fi
        """

rule summarize:
    log: join(RESULTS_DIR, "{genome}/log/{genome}.summarize.log")
    benchmark: join(RESULTS_DIR, "{genome}/log/{genome}.summarize.benchmark.txt")
    input:
        join(RESULTS_DIR, '{genome}/01.clusterseq.{genome}.tsv'),
        join(RESULTS_DIR, '{genome}/02.genotype.{genome}.tsv')
    output:
        outfile1=join(RESULTS_DIR, '{genome}/03.summarize.{genome}.clusters.tsv'),
        outfile2=join(RESULTS_DIR, '{genome}/03.summarize.{genome}.groups.tsv')
    params:
        prefix=join(RESULTS_DIR, '{genome}/03.summarize.{genome}')
    shell:
        """
        mgefinder summarize {input} -o {params.prefix} 1> {log} 2> {log}.err || \
        (cat {log}.err; exit 1)
        """


rule makefasta:
    log: join(RESULTS_DIR, "{genome}/log/{genome}.makefasta.log")
    benchmark: join(RESULTS_DIR, "{genome}/log/{genome}.makefasta.benchmark.txt")
    input:
        join(RESULTS_DIR, '{genome}/01.clusterseq.{genome}.tsv'),
        join(RESULTS_DIR, '{genome}/03.summarize.{genome}.clusters.tsv')
    output:
        outfile1=join(RESULTS_DIR, '{genome}/04.makefasta.{genome}.all_seqs.fna'),
        outfile2=join(RESULTS_DIR, '{genome}/04.makefasta.{genome}.repr_seqs.fna')
    params:
        prefix=join(RESULTS_DIR, '{genome}/04.makefasta.{genome}')
    shell:
        """
        mgefinder makefasta {input} -o {params.prefix} 1> {log} 2> {log}.err || \
        (cat {log}.err; exit 1)
        """
