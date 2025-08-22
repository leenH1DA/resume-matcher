from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import os
import json
from tkinter.filedialog import askopenfilename
import numpy
import spacy
from tkinter import *
import math
import sys
import pandas as pd
from tkinter import Tk

nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()
from collections import OrderedDict

i = -1
j = -1
k = -1
education = {}
education_set = []
soft_skill = []
softskills = []
tech_skill = []
techskills = []
index_techskill = {}
index_softskill = {}
idf_techskill = {}
idf_softskill = {}
jd_tfidf_techskill = {}
jd_tfidf_softskill = {}
tfidf_techskill = {}
tfidf_softskill = {}
cos_softskill = {}
cos_techskill = {}
cvs_dict = {}


# ==============================================================
# Preprocessing of CV docs and creating dictionary
# ==============================================================
def create_resume_index():
    DIR = 'cvs txt/'
    path = os.listdir(DIR)
    for file in path:
        input_file = DIR + file
        f1 = open(input_file, encoding="utf8")
        file_content = f1.read().lower()

        file_content = file_content.replace('/', ' ')
        file_content = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), file_content))
        file_content = ''.join(filter(lambda x: not x.isdigit(), file_content))

        f = open("Stopword-List.txt", 'r')
        for line in f:
            for st in line.split():
                file_content = file_content.lower().replace(' ' + st + ' ', ' ')
        f.close()

        doc = word_tokenize(file_content)
        doc = [stemmer.stem(t) for t in doc]

        cvs_dict[file] = doc
    # ==============================================================


# Preprocessing of JD doc
# ==============================================================
def create_jd_index(filename):
    f2 = open(filename, 'r')
    file_content = f2.read().lower()

    file_content = file_content.replace('/', ' ')
    file_content = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), file_content))
    file_content = ''.join(filter(lambda x: not x.isdigit(), file_content))

    f = open("Stopword-List.txt", 'r')
    for line in f:
        for st in line.split():
            file_content = file_content.lower().replace(' ' + st + ' ', ' ')
    f.close()

    jd_doc = word_tokenize(file_content)
    jd_doc = [stemmer.stem(t) for t in jd_doc]
    return jd_doc


def add_newruler_to_pipeline():
    data = open('skill_patterns.jsonl', 'r')
    pattern = []
    for line in data:
        sl = line.strip()
        pattern.append(json.loads(sl))
    new_ruler = nlp.add_pipe("entity_ruler")
    new_ruler.add_patterns(
        pattern)  # allows the pipeline to recognize and label entities based on the patterns defined in the file


# ===================================================================================
# Return no.of docs, names along with their tokenized resume texts extracted from doc
# ===================================================================================
def create_list():
    resume_content, resume_names = [], []
    DIR = 'cvs txt/'
    path = os.listdir(DIR)
    N = len(path)
    for file in path:
        input_file = DIR + file
        f1 = open(input_file, encoding="utf8")
        file_content = f1.read().lower()
        file_content = file_content.replace('/', ' ')
        file_content = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), file_content))

        resume_names.append(file)
        resume_content.append(nlp(file_content))
    return N, resume_content, resume_names


# ==============================================================
# Create a set of the extracted soft skills from cv
# ==============================================================
def find_resume_softskills(names, doc):
    global j
    j = j + 1
    soft_skill = set([ent.label_.upper()[7:] for ent in doc.ents if 'softsk' in ent.label_.lower()])

    temp = []
    for word in soft_skill:
        temp.append(stemmer.stem(word))
        if word not in softskills:
            softskills.append(stemmer.stem(word))

    index_softskill[names[j]] = list(temp)

    f = open("soft_skils.txt", "w")
    f.write(str(index_softskill))
    f.close()


# ==============================================================
# Create a set of the extracted technical skills from cv
# ==============================================================
def find_resume_technicalskills(names, doc):
    global i
    i = i + 1
    tech_skill = set([ent.label_.upper()[6:] for ent in doc.ents if 'skill' in ent.label_.lower()])

    temp = []
    for word in tech_skill:
        temp.append(stemmer.stem(word))
        if word not in techskills:
            techskills.append(stemmer.stem(word))

    index_techskill[names[i]] = list(temp)

    f = open("tech_skils.txt", "w")
    f.write(str(index_techskill))
    f.close()


# ==============================================================
# Create a set of the extracted education qualifications from cv
# =============================================================
def find_resume_education(doc):
    edu_set = set([ent.label_.upper()[10:] for ent in doc.ents if 'education' in ent.label_.lower()])
    return edu_set


# =======================================================================
# Create a set of the extracted soft skill from JD and calculate tf*idf
# ======================================================================
def find_jd_softskills(doc, jd_text):
    tf_softskill = {}
    temp = []

    jd_soft_skill = set([ent.label_.upper()[7:] for ent in doc.ents if 'softsk' in ent.label_.lower()])
    for word in jd_soft_skill:
        temp.append(stemmer.stem(word))
    # print(temp)
    for word in soft_skill:
        tf_softskill[word] = jd_text.count(word)

    for word in soft_skill:
        jd_tfidf_softskill[word] = tf_softskill[word] * idf_softskill[word]

    f = open("jd_tfidf_soft.txt", "w")
    f.write(str(jd_tfidf_softskill))
    f.close()


# ======================================================================
# Create a set of the extracted tech skill from JD and calculate JD tf*idf
# =====================================================================
def find_jd_techskills(doc, jd_text):
    tf_techskill = {}
    temp = []

    jd_tech_skill = set([ent.label_.upper()[6:] for ent in doc.ents if 'skill' in ent.label_.lower()])
    for word in jd_tech_skill:
        temp.append(stemmer.stem(word))
    # print(temp)
    for word in tech_skill:
        tf_techskill[word] = jd_text.count(word)

    for word in techskills:
        jd_tfidf_techskill[word] = tf_techskill[word] * idf_techskill[word]

    f = open("jd_tfidf_tech.txt", "w")
    f.write(str(jd_tfidf_techskill))
    f.close()


# =====================================================
# Calculating soft skills tf*idf
# ====================================================
def softskills_tfidf(N):
    df = {}
    tf = {}

    for w in softskills:
        if w not in soft_skill:
            soft_skill.append(w)
    # print(V_soft)

    # calculating tf
    for key in index_softskill:
        tf[key] = {}
        for w in soft_skill:
            tf[key][w] = cvs_dict[key].count(w)

            # calculating df
    for w in soft_skill:
        frq = 0
        for key in index_softskill:
            if (w in index_softskill[key]):
                frq += 1
        df[w] = frq

        # calculating idf
    for w in df:
        idf_softskill[w] = numpy.log2(N / df[w])

    # calculating tf*idf
    for key in tf:
        tfidf_softskill[key] = {}
        for w in soft_skill:
            tfidf_softskill[key][w] = tf[key][w] * idf_softskill[w]

    f = open("tfidf_soft.txt", "w")
    f.write(str(tfidf_softskill))
    f.close()


# =====================================================
# Calculating technical skills tf*idf
# ====================================================
def techskills_tfidf(N):
    df = {}
    tf = {}

    for w in techskills:
        if w not in tech_skill:
            tech_skill.append(w)
    # print(V_tech)

    # calculating tf
    for key in index_techskill:
        tf[key] = {}
        for w in tech_skill:
            tf[key][w] = cvs_dict[key].count(w)

            # calculating df
    for w in tech_skill:
        frq = 0
        for key in index_techskill:
            if (w in index_techskill[key]):
                frq += 1
        df[w] = frq

        # calculating idf
    for w in df:
        idf_techskill[w] = numpy.log2(N / df[w])

    # calculating tf*idf
    for key in tf:
        tfidf_techskill[key] = {}
        for w in tech_skill:
            tfidf_techskill[key][w] = tf[key][w] * idf_techskill[w]

    f = open("tfidf_tech.txt", "w")
    f.write(str(tfidf_techskill))
    f.close()


# =====================================================
# Calculating education qualification score
# ====================================================
def calculate_education(edu_dict):
    for key in edu_dict:
        education[key] = 0
        if 'PHD' in edu_dict[key]:
            education[key] += 0.1
        if 'POSTGRADUATE' or 'MASTERS' or 'GRADUATE' in edu_dict[key]:
            education[key] += 0.08
        if 'BACHELORS' or 'UNDERGRADUATE' in edu_dict[key]:
            education[key] += 0.06
        if 'INTERMEDIATE' in edu_dict[key]:
            education[key] += 0.04
        if 'MATRICULATION' in edu_dict[key]:
            education[key] += 0.02
        else:
            education[key] += 0

    f = open("education.txt", "w")
    f.write(str(education))
    f.close()


# ================================================================
# Computing cosine similarity
# ================================================================
def calculate_softskills_cosine_sim():
    for key in tfidf_softskill:
        dotpro = 0
        mag_cv = 0
        mag_jd = 0
        smag_jd = 0
        smag_cv = 0
        for w in jd_tfidf_softskill:
            x = tfidf_softskill[key][w] * jd_tfidf_softskill[w]
            dotpro += x
            mag_cv += (tfidf_softskill[key][w]) ** 2
            mag_jd += (jd_tfidf_softskill[w]) ** 2
        smag_cv = math.sqrt(mag_cv)
        smag_jd = math.sqrt(mag_jd)
        if smag_cv * smag_jd != 0:
            cos_softskill[key] = dotpro / (smag_cv * smag_jd)
    # print(cos_soft)


def calculate_techskills_cosine_sim():
    for key in tfidf_techskill:
        dotpro = 0
        mag_cv = 0
        mag_jd = 0
        smag_jd = 0
        smag_cv = 0
        for w in jd_tfidf_techskill:
            x = tfidf_techskill[key][w] * jd_tfidf_techskill[w]
            dotpro += x
            mag_cv += (tfidf_techskill[key][w]) ** 2
            mag_jd += (jd_tfidf_techskill[w]) ** 2
        smag_cv = math.sqrt(mag_cv)
        smag_jd = math.sqrt(mag_jd)
        if smag_cv * smag_jd != 0:
            cos_techskill[key] = dotpro / (smag_cv * smag_jd)
    # print(cos_tech)


def show_result():
    total = {}
    for key in cos_techskill:
        if key in cos_softskill:
            total[key] = cos_softskill[key] + cos_techskill[key] + education[key]
    # print(total)

    sorted_total = OrderedDict(sorted(total.items(), key=lambda x: x[1], reverse=True))
    f = open("cosine_simlarity", "w")
    f.write(str(sorted_total))
    f.close()

    final = {}
    alpha = 0.05
    for key in sorted_total:
        if (sorted_total[key] >= alpha):
            final[key] = sorted_total[key]
    accepted = []
    c = 0
    for key in final:
        accepted.append([])
        accepted[c].append(key)
        accepted[c].append(final[key])
        c = c + 1
    count = len(accepted)
    l = []
    for j in range(count):
        l.append(j + 1)

    root = Tk()
    root.geometry('660x400')
    root.title("Matching CV-JD")
    label = ['Accepted Resumes', 'Scores']
    txt = Text(root)
    txt.pack()

    class PrintToTXT(object):
        def write(self, s):
            txt.insert(END, s)

    sys.stdout = PrintToTXT()
    dframe = pd.DataFrame(accepted, index=l, columns=label)
    print(dframe)
    mainloop()


# =================================================
# -------------main
# =================================================
create_resume_index()
add_newruler_to_pipeline()

N, resume_texts, resume_names = create_list()

for text in resume_texts:
    find_resume_softskills(resume_names, text)
    find_resume_technicalskills(resume_names, text)
    education_set.append(find_resume_education(text))

edu_dict = dict(zip(resume_names, education_set))
calculate_education(edu_dict)

softskills_tfidf(N)
techskills_tfidf(N)

filename = askopenfilename()  # Select JD file
print(filename)
doc = create_jd_index(filename)

f2 = open(filename, 'r')
jd_text = f2.read().lower()
jd_text = nlp(jd_text)

find_jd_softskills(jd_text, doc)
find_jd_techskills(jd_text, doc)
calculate_softskills_cosine_sim()
calculate_techskills_cosine_sim()
Tk().withdraw()
show_result()
